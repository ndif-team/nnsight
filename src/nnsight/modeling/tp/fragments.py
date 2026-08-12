"""Show intervention code whole tensors on a tensor-parallel model.

Under transformers tensor parallelism a module's activation can be one rank's
slice rather than the real thing: a column-parallel linear splits its *output*
across ranks, and a row-parallel linear takes its *input* already split. A user
asked for the layer, not a quarter of it, so those values are gathered before a
worker sees them and re-split before the model's own forward carries on.

Two facts make this cheap to arrange.

**transformers labels the shards for us.** ``apply_tensor_parallelism`` stamps
every module it shards with ``_hf_tp_plan`` (the style name) and
``_hf_device_mesh``, so which side of which module carries a shard is read off
the module, not guessed. These rules are handed every envoy through
``instrument`` as the tree is built, so they record themselves — and learn whether
there is anything to do at all — right there.

**nnsight already sees the post-TP value.** transformers registers its own
forward hooks at load; nnsight registers the interleaver's when the
[`Envoy`][nnsight.intervention.envoy.Envoy] tree is built, i.e. afterwards. Hooks
fire in registration order, so by the time
[`TPFragments.whole`][nnsight.modeling.tp.fragments.TPFragments.whole]
runs, a row-parallel output has already been all-reduced and a
``colwise_gather_output`` head already all-gathered — those arrive whole and are
left alone. Only the genuinely sharded sides are listed in
[`SHARDED_SIDES`][nnsight.modeling.tp.fragments.SHARDED_SIDES].

Every rank runs the same intervention block, so every rank reaches ``handle`` at
the same location with the same parked workers and makes the same decision to
gather. That is what keeps the collectives matched; it is also why a block whose
control flow diverges across ranks deadlocks, and why nothing here may branch on
rank. For the same reason a run whose *sampling* diverges is not merely
inconsistent but wrong — the ranks would go on to all-reduce activations computed
from different tokens — so seed every rank identically before generating.

Everything about *when* to gather — once per visit, only when something is
waiting, re-split on the way out — belongs to
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle] and is
shared with every other distributed runtime. See
[`nnsight.intervention.fragments`][nnsight.intervention.fragments] for that half,
including why it is the interleaver's job and not the batcher's.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import torch

from ...intervention.fragments import Fragments
from ...util import apply

# Which side of a module carries this rank's slice, per transformers TP style.
# A side listed here is sharded along the LAST dim and must be gathered before a
# worker sees it; a side left out is already whole, because the style's own
# collective (an all-reduce, or an all-gather for `gather_output`) ran in the TP
# hook that fires before ours.
#
# Starred entries were measured on Llama-3.2-3B at tp=4: the sharded sides came
# back at 1/4 width and reassembled with a rank-order `cat(-1)`, and the whole
# sides were bit-identical across ranks. The rest follow from
# transformers/integrations/tensor_parallel.py but no model in the test set
# exercised them — treat them as unverified.
SHARDED_SIDES: Dict[str, Tuple[str, ...]] = {
    "colwise": ("output",),                   # * output features split
    "packed_colwise": ("output",),            #   (fused gate_up, same split)
    "colwise_gather_output": (),              # * gather_output=True -> whole
    "rowwise": ("input",),                    # * input pre-split; output all-reduced
    "rowwise_split_input": ("input",),        #   TP splits it in its own pre-hook
    "packed_rowwise": ("input",),
    "embedding_rowwise": (),                  # * vocab-parallel; output all-reduced
    "embedding_colwise": (),                  # + output all-reduced (see below)
    "sequence_parallel": ("output",),         # + reduce-scatter, on the LAST dim
    "all_reduce": (),                         #   output all-reduced -> whole
    "replicated_with_grad_allreduce": (),     #   params replicated; activations whole
    "moe_tp_experts": (),                     # + output all-reduced (see below)
}

# Entries marked + were settled by reading transformers/integrations/
# tensor_parallel.py rather than by running a model, because no checkpoint in the
# test set exercises them. Each is a case where the obvious reading of the style's
# *name* is wrong:
#
# * ``embedding_colwise`` splits the hidden dim, so its output looks like a
#   fragment — but ``EmbeddingParallel._prepare_output_fn`` ends in an
#   unconditional ``all_reduce_forward``; the ``embedding_dim_sharding == 0``
#   branch above it guards only the vocab masking. Listing "output" here would
#   all-gather an already-whole tensor and handed users one ``tp_size`` times too
#   wide with a plausible first copy — the same failure as the tied-LM-head bug
#   MINIMUM_TRANSFORMERS exists for, reintroduced by this table.
# * ``moe_tp_experts`` is expert-parallel and was previously refused outright.
#   Its forward needs nothing: ``_prepare_input_fn`` applies only
#   ``all_reduce_backward`` (identity going forwards) and ``_prepare_output_fn``
#   is ``all_reduce_forward``. Both sides are whole, exactly like ``all_reduce``.
#   Refusing it cost every MoE checkpoint that uses it — 26 of the configs
#   shipped with transformers 5.15, including Mixtral, DeepSeek-V3 and Qwen3-MoE,
#   were refused for this and nothing else.
# * ``sequence_parallel`` names a sequence dim and takes a ``sequence_dim=1``
#   argument, which it stores and never uses: its reduce-scatter hardcodes
#   ``x.dim() - 1``. The entry matches the implementation and contradicts the
#   declared intent, so it is the one most likely to rot — see the shape check in
#   `_gather_whole`, which is what would catch it.

# Styles refused rather than guessed at, with the reason a user is shown. These
# slice something other than the last dim — by expert, or into a fused kv
# projection — so neither the gather nor the re-split above means anything for
# them. A model using one fails at load rather than silently handing users a
# fragment.
#
# Being on this list is a claim that the style *needs* a rule nnsight doesn't
# have, not merely that no one has checked it. ``moe_tp_experts`` was here on the
# strength of its name and did not belong: it is expert-parallel and still needs
# nothing, because its forward all-reduces (see SHARDED_SIDES). Prefer reading
# the style's ``_prepare_input_fn``/``_prepare_output_fn`` to inferring from the
# family it belongs to — the two do not track each other.
UNSUPPORTED: Dict[str, str] = {
    "grouped_gemm": "expert-parallel (MoE)",
    "ep_router": "expert-parallel (MoE)",
    "megamoe_router": "expert-parallel (MoE)",
    "megamoe_experts": "expert-parallel (MoE)",
    "moe_identity_expert": "expert-parallel (MoE)",
    "mla_kv_a_proj": "MLA split kv projection",
}

# The oldest transformers whose tensor parallelism produces correct activations.
#
# 5.14.1 shards a tied LM head's *hook* but not its weight, so on a checkpoint
# with `tie_word_embeddings=True` the head keeps its full weight while
# `colwise_gather_output` all-gathers the result anyway: logits come back
# `tp_size` times too wide, inside transformers, before nnsight sees anything.
# Nothing downstream looks wrong — the argmax still lands inside the first copy —
# so it survives every casual check. Measured on Llama-3.2-3B at tp=4: width
# 513024 against a vocabulary of 128256 on 5.14.1, correct on 5.15.0.
MINIMUM_TRANSFORMERS = "5.15.0"


class UnsupportedTransformersVersion(RuntimeError):
    """transformers is too old to shard a model correctly."""


def _check_transformers_version() -> None:
    """Refuse to trace a sharded model on a transformers known to mis-shard.

    Called once, the first time a genuinely sharded module is seen, so an
    unsharded model on an old transformers is unaffected.
    """
    global _version_checked
    if _version_checked:
        return
    _version_checked = True

    import transformers
    from packaging.version import Version

    installed = transformers.__version__
    # Compared on the release numbers alone, so a `5.15.0.dev0` — which a plain
    # `>=` sorts *below* 5.15.0, pre-releases coming first — counts as 5.15. A
    # dev build of the fixed series is the normal way to be running it early;
    # refusing those would make an editable transformers checkout unusable.
    if Version(installed).release >= Version(MINIMUM_TRANSFORMERS).release:
        return

    raise UnsupportedTransformersVersion(
        f"tensor parallelism needs transformers >= {MINIMUM_TRANSFORMERS}, but "
        f"{installed} is installed. Older versions do not shard a tied LM head's "
        "weight while still gathering its output, so a model with "
        "`tie_word_embeddings=True` returns logits `tp_size` times too wide — "
        "wrong in a way that still produces a plausible argmax. Upgrade "
        "transformers, or load this model on one GPU."
    )


_version_checked = False


class UnsupportedParallelStyle(Exception):
    """The model shards something interventions can't be shown whole."""


def _shardable(tensor: torch.Tensor, world_size: int) -> bool:
    """Whether ``tensor`` can be this rank's slice of a last-dim shard.

    The sharded activation is always a float tensor whose last dim divides the
    mesh — transformers' own all-gather assumes equal shards. Anything else (an
    integer mask, a scalar, a ragged width) cannot be what was split, so it
    passes through untouched rather than being corrupted by a collective.
    """
    return (
        tensor.is_floating_point()
        and tensor.dim() >= 1
        and tensor.shape[-1] % world_size == 0
    )


def _gather(value: Any, mesh: Any) -> Any:
    """Every rank's slice of ``value``, concatenated back into the whole tensor."""
    from transformers.integrations.tensor_parallel import all_gather

    world_size = mesh.size()
    return apply(
        value,
        lambda tensor: (
            all_gather(tensor, mesh) if _shardable(tensor, world_size) else tensor
        ),
        torch.Tensor,
    )


def _last_dim(value: Any) -> Optional[int]:
    """The last dimension of the one shardable tensor in ``value``, if there is one.

    A location's value is often a tuple or a dataclass, most of whose members
    are not fragments — a mask, a cache, ``None``. Exactly one shardable tensor
    makes the width comparable before and after a gather; anything else (none, or
    several of different widths) is not something to draw a conclusion from, and
    the caller skips the check rather than guessing.
    """
    widths = []
    apply(
        value,
        lambda tensor: widths.append(tensor.shape[-1])
        if tensor.is_floating_point() and tensor.dim() >= 1
        else None,
        torch.Tensor,
    )
    return widths[0] if len(widths) == 1 else None


def _reshard(value: Any, mesh: Any) -> Any:
    """This rank's slice of ``value``, as the model's own forward expects it.

    transformers' ``split`` is the exact inverse of its ``all_gather`` — chunk
    along the last dim, take this rank's — so a value no worker touched comes back
    unchanged, and an edited one carries the edit. Both are autograd functions, so
    the pair is transparent to a backward pass as well.

    Guarded by the same predicate as the gather, and it must be: what goes back
    is what the model's own forward consumes. A tensor ``_gather`` declined to
    touch was never a fragment, so splitting it here hands the model a slice of
    something whole. Divisibility alone is not enough to tell those apart — an
    integer ``position_ids`` of width 8 at tp=4 divides perfectly, and every rank
    would silently continue on a quarter of it. Wrong answers, not a hang.
    """
    from transformers.integrations.tensor_parallel import split

    world_size = mesh.size()
    return apply(
        value,
        lambda tensor: (
            split(tensor, mesh) if _shardable(tensor, world_size) else tensor
        ),
        torch.Tensor,
    )


def is_sharded(module: torch.nn.Module) -> bool:
    """Whether ``module``'s tree has anything split across a multi-rank mesh.

    A caller's way to ask whether a loaded model is sharded at all, without
    reaching into transformers' stamps itself. Nothing in nnsight branches on it
    — a [`TPFragments`][nnsight.modeling.tp.fragments.TPFragments] is built
    for every HuggingFace model and stays inert until
    [`instrument`][nnsight.modeling.tp.fragments.TPFragments.instrument] finds
    a sharded module — but a server deciding how to run a replica wants it.

    A degenerate 1-rank mesh reads as not sharded: there is nothing to gather.
    """
    for child in module.modules():
        if getattr(child, "_hf_tp_plan", None) is None:
            continue
        mesh = getattr(child, "_hf_device_mesh", None)
        if mesh is not None and mesh.size() > 1:
            return True
    return False


class TPFragments(Fragments):
    """Which values a transformers-sharded model splits, and how to reassemble them.

    A [`HuggingFaceModel`][nnsight.modeling.huggingface.HuggingFaceModel] is built
    with one of these whether or not it is sharded, because whether it *is* only
    becomes knowable as the model loads. It starts
    [`enabled`][nnsight.intervention.fragments.Fragments.enabled]``=False`` and
    does nothing at all until
    [`instrument`][nnsight.modeling.tp.fragments.TPFragments.instrument] finds
    something actually split across ranks.

    Attributes:
        enabled: Whether anything in this tree is sharded. False costs one
            attribute check per handled location.
        tp_rules: Sharded location -> the device mesh it is split over. A location
            absent from it is already whole.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.tp_rules: Dict[str, Any] = {}
        # Source locations already warned about; see `read`.
        self._warned: set = set()
        # Locations whose gather has been shape-checked; see `whole`.
        # The rule for a location is a property of the loaded model, so
        # re-checking it every trace buys nothing.
        self._checked: set = set()

    def instrument(self, envoy: Any) -> None:
        """Install the hooks, and record whether this envoy's value is a shard.

        Called once per envoy as the tree is built, and again through
        [`_update`][nnsight.intervention.envoy.Envoy._update] when real weights are
        dispatched under a tree built on meta. That is the one moment both the
        module — carrying transformers' ``_hf_tp_plan`` — and its path are in
        hand, and it covers both load paths without either having to say so: a
        meta module carries no plan and registers nothing, and the same envoy
        re-instrumented over real weights registers then.

        Raises:
            UnsupportedParallelStyle: if this module is sharded in a way there is
                no rule for. Refused as the tree is built rather than papered
                over, since the alternative is silently handing users a fragment
                of a tensor.
        """
        super().instrument(envoy)

        module = envoy._module
        style = getattr(module, "_hf_tp_plan", None)
        if style is None:
            return

        if style in UNSUPPORTED:
            raise UnsupportedParallelStyle(
                f"'{envoy.path}' is sharded as '{style}' ({UNSUPPORTED[style]}), "
                "which interventions can't be shown whole, so this model can't "
                "be traced tensor-parallel."
            )
        if style not in SHARDED_SIDES:
            raise UnsupportedParallelStyle(
                f"'{envoy.path}' is sharded as '{style}', which is not a parallel "
                "style this version of nnsight recognizes."
            )

        mesh = getattr(module, "_hf_device_mesh", None)
        if mesh is None or mesh.size() == 1:
            # Stamped but not actually split (a degenerate 1-rank mesh) — nothing
            # to gather, and a collective over a 1-rank group is pure overhead.
            return

        # The first genuinely sharded module: past here the weights are already
        # split, so this is the last point at which refusing is still cheaper
        # than returning wrong numbers.
        _check_transformers_version()

        self.enabled = True
        for side in SHARDED_SIDES[style]:
            self.tp_rules[f"{envoy.path}.{side}"] = mesh

    def begin(self) -> None:
        # A fresh run warns again about anything it reads. A model actor serves
        # request after request from one process, and the second user deserves
        # the same caveat as the first.
        self._warned.clear()

    def fragmented(self, location: str) -> bool:
        """Whether this location's value is one rank's slice.

        A dict lookup: the rules were recorded at instrument time, so nothing is
        inspected here and nothing branches on rank.
        """
        return location in self.tp_rules

    def read(self, location: str) -> None:
        """Warn that a ``.source`` read is this rank's shard, and hand it over.

        `.source` exposes a module's *intermediate* values, and the rules here
        describe module *boundaries* — a different question. Measured at tp=2:
        ``mlp.output`` and ``gate_proj.output`` arrive whole, while
        ``mlp.source.self_gate_proj_0`` comes back at half width.

        These cannot be gathered for the reader. Knowing a value is sharded is
        not enough to reassemble it: which axis it is split on changes through a
        module's forward — a fused qkv projection is split by head on the way out
        of the linear and by hidden dim after the reshape — and the axis is not
        recoverable from the tensor at runtime. So the value is handed over as-is
        with a warning, rather than refused (a `.source` read is often exactly
        what someone debugging a sharded model wants) or silently corrected.

        **Do not branch on one.** This is the only rank-dependent value nnsight
        ever gives intervention code, and every rank must reach the same
        collectives in the same order. ``if x.source.self_k_proj_0.output.max() >
        t:`` takes different branches on different ranks and deadlocks the group.
        """
        if ".source." not in location or location in self._warned:
            return

        self._warned.add(location)
        # This module's registry entry is cleared first because a model actor
        # serves request after request from one process: Python's default filter
        # shows a warning once per source line, so without this only the first
        # user of a replica would ever see it.
        globals().pop("__warningregistry__", None)
        warnings.warn(
            f"'{location}' reads inside a module's forward on a model split "
            "across ranks. `.source` values can be this rank's shard, and which "
            "axis they are split on changes through the forward, so they are "
            "handed over as-is rather than gathered. A value past the layer that "
            "all-reduces (a module's own `.input`/`.output`) is whole; one "
            "between a column-parallel layer and it is not. Do not branch on one "
            "— the ranks would diverge. Compare against a single-GPU run if it "
            "matters.",
            UserWarning,
            stacklevel=2,
        )

    def whole(self, location: str, value: Any) -> Any:
        """Every rank's slice of ``value``, concatenated into the real tensor."""
        return self._gather_whole(location, value, self.tp_rules[location])

    def fragment(self, location: str, whole: Any) -> Any:
        """This rank's slice again, carrying whatever the workers left behind."""
        return _reshard(whole, self.tp_rules[location])

    def _gather_whole(self, provider: str, value: Any, mesh: Any) -> Any:
        """``value`` gathered, checked once per location against the rule used.

        The rules in [`SHARDED_SIDES`][nnsight.modeling.tp.fragments.SHARDED_SIDES]
        are a claim about a transformers version, and most of them were settled by
        reading its source rather than by running a model. This is what turns a
        wrong claim from silent into loud.

        The check is cheap and total: a side listed as sharded must actually grow
        by ``world_size`` when gathered. A rule that named a whole value leaves
        the width unchanged after the collective — which is precisely the shape of
        the failure that motivated
        [`MINIMUM_TRANSFORMERS`][nnsight.modeling.tp.fragments.MINIMUM_TRANSFORMERS]:
        a value already made whole by the model's own hook, gathered a second time
        and handed back ``tp_size`` times too wide with a plausible first copy.
        A version floor catches that for one known version; this catches it for
        every version, including the one after next.

        Once per location per run, not per visit — the answer is a property of
        the rule, and a generation loop revisits the same location hundreds of
        times.

        Raises:
            UnsupportedParallelStyle: if the gather did not widen the value.
        """
        gathered = _gather(value, mesh)

        if provider in self._checked:
            return gathered
        self._checked.add(provider)

        before, after = _last_dim(value), _last_dim(gathered)
        if before is None or after is None:
            return gathered

        expected = before * mesh.size()
        if after != expected:
            raise UnsupportedParallelStyle(
                f"'{provider}' is listed as carrying this rank's shard, but "
                f"gathering it across {mesh.size()} ranks gave a last dimension "
                f"of {after} where {expected} was expected (it was {before} "
                "before). Either this transformers version already makes the "
                "value whole — in which case gathering it hands out a tensor "
                f"{mesh.size()} times too wide — or it is split on an axis these "
                "rules don't describe. Refusing rather than returning it."
            )
        return gathered


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
the module, not guessed. The interleaver is handed every envoy through
``instrument`` as the tree is built, so it records its own rules — and learns
whether it has anything to do at all — right there.

**nnsight already sees the post-TP value.** transformers registers its own
forward hooks at load; nnsight registers the interleaver's when the
[`Envoy`][nnsight.intervention.envoy.Envoy] tree is built, i.e. afterwards. Hooks
fire in registration order, so by the time
[`TPInterleaver.handle`][nnsight.modeling.tp.interleaver.TPInterleaver.handle]
runs, a row-parallel output has already been all-reduced and a
``colwise_gather_output`` head already all-gathered — those arrive whole and are
left alone. Only the genuinely sharded sides are listed in
[`SHARDED_SIDES`][nnsight.modeling.tp.interleaver.SHARDED_SIDES].

Every rank runs the same intervention block, so every rank reaches ``handle`` at
the same location with the same parked workers and makes the same decision to
gather. That is what keeps the collectives matched; it is also why a block whose
control flow diverges across ranks deadlocks, and why nothing here may branch on
rank. For the same reason a run whose *sampling* diverges is not merely
inconsistent but wrong — the ranks would go on to all-reduce activations computed
from different tokens — so seed every rank identically before generating.

## Why the interleaver and not the batcher

vLLM does the equivalent gather in a
[`Batcher`][nnsight.intervention.batching.Batcher] subclass
([`VLLMBatcher`][nnsight.modeling.vllm.batching.VLLMBatcher]), because it is
already overriding the batcher for vLLM's flat token axis, and because its model
runner constructs the batcher itself and assigns ``interleaver.batcher`` directly
— no trace is deserialized in the worker, only a
[`Mediator`][nnsight.intervention.interleaver.Mediator].

That does not carry over to a deserialized *trace*, which is how a remote request
runs. ``Envoy.__getstate__`` pickles the envoy **by value**, tagging only its
module and its interleaver as persistent ids. So the tracer executing server-side
holds a *copy of the client's* envoy, and
``self.batcher = self.envoy._batcher_class(...)`` resolves against the client's
class — while ``self.envoy.interleaver`` resolves, through the persistent id, to
the **server's own** interleaver. Only one of those two is an object the serving
process controls.

Sitting on the interleaver is also the better fit on its own terms: the
collective fires once per location per visit however many workers read it (no
per-worker memoization needed), and it composes with the real batcher, whose
dim-0 row narrowing happens inside this bracket.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import torch

from ...intervention.interleaver import Interleaver
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
    "embedding_colwise": ("output",),         #   hidden dim split
    "sequence_parallel": ("output",),         #   reduce-scatter on the last dim
    "all_reduce": (),                         #   output all-reduced -> whole
    "replicated_with_grad_allreduce": (),     #   params replicated; activations whole
}

# Styles refused rather than guessed at. The expert-parallel family slices by
# *expert*, not along the last dim, so neither the gather nor the re-split below
# is meaningful for it; MLA's split kv projection needs its own rule. A model
# that uses one of these fails at load instead of silently handing users a
# fragment of a tensor.
UNSUPPORTED: Dict[str, str] = {
    "grouped_gemm": "expert-parallel (MoE)",
    "ep_router": "expert-parallel (MoE)",
    "megamoe_router": "expert-parallel (MoE)",
    "moe_tp_experts": "expert-parallel (MoE)",
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


def _reshard(value: Any, mesh: Any) -> Any:
    """This rank's slice of ``value``, as the model's own forward expects it.

    transformers' ``split`` is the exact inverse of its ``all_gather`` — chunk
    along the last dim, take this rank's — so a value no worker touched comes back
    unchanged, and an edited one carries the edit. Both are autograd functions, so
    the pair is transparent to a backward pass as well.
    """
    from transformers.integrations.tensor_parallel import split

    world_size = mesh.size()
    return apply(
        value,
        lambda tensor: (
            split(tensor, mesh) if tensor.shape[-1] % world_size == 0 else tensor
        ),
        torch.Tensor,
    )


def is_sharded(module: torch.nn.Module) -> bool:
    """Whether ``module``'s tree has anything split across a multi-rank mesh.

    The test [`Envoy._interleaver_class`][nnsight.intervention.envoy.Envoy._interleaver_class]
    uses to decide whether a model needs a
    [`TPInterleaver`][nnsight.modeling.tp.interleaver.TPInterleaver] at all. A
    degenerate 1-rank mesh reads as not sharded: there is nothing to gather.
    """
    for child in module.modules():
        if getattr(child, "_hf_tp_plan", None) is None:
            continue
        mesh = getattr(child, "_hf_device_mesh", None)
        if mesh is not None and mesh.size() > 1:
            return True
    return False


class TPInterleaver(Interleaver):
    """An [`Interleaver`][nnsight.intervention.interleaver.Interleaver] that hands
    workers whole tensors on a tensor-parallel model.

    A [`HuggingFaceModel`][nnsight.modeling.huggingface.HuggingFaceModel] is built
    with one of these whether or not it is sharded, because whether it *is* only
    becomes knowable as the model loads. It starts
    [`enabled`][nnsight.modeling.tp.interleaver.TPInterleaver.enabled]``=False``
    and behaves exactly like the base interleaver until
    [`instrument`][nnsight.modeling.tp.interleaver.TPInterleaver.instrument] finds
    something actually split across ranks.

    Attributes:
        enabled: Whether anything in this tree is sharded. False costs one
            attribute check per handled location.
        tp_rules: Sharded location -> the device mesh it is split over. A location
            absent from it is already whole.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False
        self.tp_rules: Dict[str, Any] = {}
        # Source locations already warned about this run; see `_warn_source`.
        self._warned: set = set()

    def __enter__(self) -> "TPInterleaver":
        # A fresh run: warn again about anything it reads (see `_warn_source`).
        self._warned.clear()
        return super().__enter__()

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

    def handle(self, provider: str, value: Any) -> Any:
        """Gather ``value`` if it is a shard someone is waiting for, serve the
        workers, then hand the model back its own slice.

        On an unsharded model this is the base interleaver plus one attribute
        check. Otherwise the gather is still skipped unless a worker is actually
        parked on this visit
        ([`observed`][nnsight.modeling.tp.interleaver.TPInterleaver.observed]), so
        an untouched sharded location costs nothing — the common case, since a
        trace reads a handful of locations out of hundreds.
        """
        if not self.enabled:
            return super().handle(provider, value)

        self._warn_source(provider)

        mesh = self.tp_rules.get(provider)
        if mesh is None or not self.observed(provider):
            return super().handle(provider, value)

        whole = super().handle(provider, _gather(value, mesh))
        return _reshard(whole, mesh)

    def _warn_source(self, provider: str) -> None:
        """Warn that a ``.source`` read is this rank's shard, and hand it over.

        `.source` exposes a module's *intermediate* values, and the rules here
        describe module *boundaries* — a different question. Measured at tp=2:
        ``mlp.output`` and ``gate_proj.output`` arrive whole, while
        ``mlp.source.self_gate_proj_0`` comes back at half width.

        These cannot be gathered for the reader. Knowing a value is sharded is
        easy — everything between a column-parallel output and the row-parallel
        layer that all-reduces it is — but gathering needs the **axis**, and the
        axis moves within that window as attention reshapes: measured at ``-1``
        after ``k_proj``, ``2`` after ``view``, ``1`` after ``transpose``, back to
        ``-1`` after the output reshape. Every rank holds the same shape, so
        nothing at runtime recovers it.

        A warning rather than an error because plenty of source reads under a
        sharded model are perfectly whole — anything past the row-parallel layer
        that closed the window (``mlp.source.self_down_proj_0`` measures 16 at
        both tp=1 and tp=2), and anything in a part of the model that shards
        nothing. Refusing the lot to catch the sharded ones blocks correct work;
        saying so and handing the value over does not.

        Warned once per location per run: a read inside a generation loop fires
        on every token, and a caveat repeated hundreds of times is one nobody
        reads either.
        """
        if ".source." not in provider or provider in self._warned:
            return
        if not self.observed(provider):
            return

        self._warned.add(provider)
        # This module's registry entry is cleared first because a model actor
        # serves request after request from one process: Python's default filter
        # shows a warning once per source line, so without this only the first
        # user of a replica would ever see it.
        globals().pop("__warningregistry__", None)
        warnings.warn(
            f"'{provider}' reads inside a module's forward on a model split "
            "across ranks. `.source` values can be this rank's shard, and which "
            "axis they are split on changes through the forward, so they are "
            "handed over as-is rather than gathered. A value past the layer that "
            "all-reduces (a module's own `.input`/`.output`) is whole; one "
            "between a column-parallel layer and it is not. Compare against a "
            "single-GPU run if it matters.",
            stacklevel=2,
        )

    def observed(self, provider: str) -> bool:
        """Whether any worker is waiting on *this* visit to ``provider``.

        Mirrors the match [`Mediator.handle`][nnsight.intervention.interleaver.Mediator.handle]
        makes — a worker parks already carrying the occurrence tag it wants, so
        this is the same single string comparison against this visit's count.
        Deliberately not "is any worker parked anywhere": a worker waiting on a
        *later* iteration of this location must not trigger the collective now.

        A ``tracer.cache()`` counts too — it would otherwise record fragments —
        but only for the locations it actually keeps. A cache is usually scoped to
        a handful of modules, so asking it (rather than assuming any open cache
        wants everything) is the difference between a few collectives and one at
        every sharded module in the model.

        Every rank evaluates this over the same block, so it answers the same
        everywhere — which is what keeps the ranks' collectives matched.
        """
        for mediator in self.mediators:
            pending = mediator.pending
            if (
                pending is not None
                and pending[1] == f"{provider}.i{mediator.iterations[provider]}"
            ):
                return True
            if any(cache.wants(provider) for cache in mediator.caches):
                return True
        return False

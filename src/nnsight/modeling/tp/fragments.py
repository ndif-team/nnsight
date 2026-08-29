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

**nnsight sees the pre-collective value.** The interleaver's handoff runs inside
a module's forward — after transformers' pre-hook, before its post-hook — so a
row-parallel output is still this rank's *partial sum* there, and a
``colwise_gather_output`` head still a shard. [`SIDES`][nnsight.modeling.tp.fragments.SIDES]
says which, per style, and [`TPFragments.whole`][nnsight.modeling.tp.fragments.TPFragments.whole]
all-gathers a shard or all-reduces a partial. What goes back is chosen so the
module's own post-hook completes the picture: a shard's slice, or — for a
partial — the whole on rank 0 and zeros elsewhere, which the post-hook's reduce
turns into exactly the (possibly edited) whole on every rank.

The rules describe module *boundaries*. A ``.source`` value between two ops
inside a forward can be a shard split on an axis that moves through the forward,
so it is handed over as-is: whole past the layer that all-reduces, this rank's
slice between a column-parallel layer and it. Compare against a single-GPU run
if it matters, and never branch on one — the ranks would diverge.

Every rank runs the same intervention block, so every rank reaches ``handle`` at
the same location with the same parked workers and makes the same decision to
gather — which is what keeps the collectives matched, why a block whose control
flow diverges across ranks deadlocks, and why sampling must be seeded identically
on every rank.

Everything about *when* to gather — once per visit, only when something is
waiting, re-split on the way out — belongs to
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle] and is
shared with every other distributed runtime. See
[`nnsight.intervention.fragments`][nnsight.intervention.fragments] for that half,
including why it is the interleaver's job and not the batcher's.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ...intervention.fragments import Fragments
from ...util import apply

# What each side of a module is at the interleaver's handoff, per transformers
# TP style. The handoff runs inside the module's forward — after the style's
# pre-hook, before its post-hook — so a side is one of:
#
#   "shard"    this rank's slice along the LAST dim; made whole by an all-gather
#              and handed back as this rank's slice again.
#   "partial"  this rank's term of a sum the style's post-hook will all-reduce
#              (or reduce-scatter); made whole by an all-reduce and handed back as
#              the whole on rank 0 and zeros elsewhere, so the post-hook's own
#              reduce yields exactly the (possibly edited) whole on every rank.
#
# A side left out is already whole there. Starred entries were measured on
# Llama-3.2-3B at tp=4; the rest follow from
# transformers/integrations/tensor_parallel.py but no model in the test set
# exercised them — treat them as unverified.
SIDES: Dict[str, Dict[str, str]] = {
    "colwise": {"output": "shard"},                       # * output features split
    "packed_colwise": {"output": "shard"},                #   (fused gate_up, same split)
    "colwise_gather_output": {"output": "shard"},         # * gathered by its post-hook, after us
    "rowwise": {"input": "shard", "output": "partial"},   # * input pre-split; post-hook all-reduces
    "rowwise_split_input": {"input": "shard", "output": "partial"},  # pre-hook splits the input
    "packed_rowwise": {"input": "shard", "output": "partial"},
    "embedding_rowwise": {"output": "partial"},           # * vocab-parallel; post-hook all-reduces
    "embedding_colwise": {"output": "partial"},           # + post-hook all-reduces (see below)
    "sequence_parallel": {"output": "partial"},           # + post-hook reduce-scatters, on the LAST dim
    "all_reduce": {"output": "partial"},
    "replicated_with_grad_allreduce": {},                 #   params replicated; activations whole
    "moe_tp_experts": {"output": "partial"},              # + post-hook all-reduces (see below)
}

# Entries marked + follow from reading transformers/integrations/tensor_parallel.py
# rather than from running a model: ``embedding_colwise`` and ``moe_tp_experts``
# end in an unconditional ``all_reduce_forward`` post-hook (so a partial, not the
# shard its name suggests), and ``sequence_parallel`` reduce-scatters on the last
# dim in its post-hook (whole-width partial at the handoff).

# Styles refused rather than guessed at, with the reason a user is shown: these
# slice something other than the last dim — by expert, or into a fused kv
# projection — so neither the gather nor the re-split above means anything for
# them. Read the style's ``_prepare_input_fn``/``_prepare_output_fn`` before
# adding one; the name alone misled on ``moe_tp_experts``.
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

    A sharded activation is a float tensor with a last dim; anything else (an
    integer mask, a scalar) cannot be what was split, so it passes through
    untouched rather than being corrupted by a collective. Only the whole being
    split back has to chunk evenly — a shard's own width need not divide.
    """
    return tensor.is_floating_point() and tensor.dim() >= 1 and tensor.shape[-1] % world_size == 0


def _collectives():
    """transformers' ``(all_gather, split)``, the pair fragments are moved with.

    Imported through here rather than at each call site so a transformers that
    does not have them is one error, raised where it can be explained.

    5.16.0 rebuilt tensor parallelism on DTensor: the implementation moved to
    ``transformers.distributed.tensor_parallel``, and these two module-level
    helpers are not part of what the old path re-exports -- they are gone, with
    no drop-in replacement (a DTensor is made whole through
    ``redistribute``/``full_tensor``). Left alone, the failure is an ImportError
    from inside a hook on the first sharded module, naming a module that does
    still import, on a code path that only runs on multiple GPUs.
    """
    try:
        from transformers.integrations.tensor_parallel import all_gather, split
    except ImportError as error:
        import transformers

        raise UnsupportedTransformersVersion(
            f"transformers {transformers.__version__} does not provide the "
            "`all_gather` / `split` helpers that nnsight reassembles sharded "
            "activations with, so a fragment cannot be made whole for an "
            "intervention to see. 5.16.0 rebuilt tensor parallelism on DTensor "
            "and removed them. Pin `transformers<5.16`, or load this model on "
            "one GPU."
        ) from error

    return all_gather, split


def _gather(value: Any, mesh: Any) -> Any:
    """Every rank's slice of ``value``, concatenated back into the whole tensor."""
    all_gather, _ = _collectives()

    return apply(
        value,
        lambda tensor: all_gather(tensor, mesh) if tensor.is_floating_point() and tensor.dim() else tensor,
        torch.Tensor,
    )


def _reduce(value: Any, mesh: Any) -> Any:
    """Every rank's partial of ``value`` summed — the tensor the post-hook would make."""
    import torch.distributed as dist

    group = mesh.get_group()

    def summed(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            return tensor
        total = tensor.clone()
        dist.all_reduce(total, group=group)
        return total

    return apply(value, summed, torch.Tensor)


def _reshard(value: Any, mesh: Any) -> Any:
    """This rank's slice of ``value``, as the model's own forward expects it.

    transformers' ``split`` is the exact inverse of its ``all_gather``, and both
    are autograd functions. Guarded by the same predicate as the gather: a tensor
    that was never a fragment must not be cut down here.
    """
    _, split = _collectives()

    world_size = mesh.size()
    return apply(
        value,
        lambda tensor: (
            split(tensor, mesh) if _shardable(tensor, world_size) else tensor
        ),
        torch.Tensor,
    )


class TPFragments(Fragments):
    """Which values a transformers-sharded model splits, and how to reassemble them.

    Built for every HuggingFace model and inert (``enabled=False``) until
    `instrument` finds a module actually split across ranks.

    Attributes:
        enabled: Whether anything in this tree is sharded.
        tp_rules: Location -> ``(mesh, kind)``, ``kind`` being ``"shard"`` or
            ``"partial"`` (see `SIDES`). A location absent from it is already whole.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.tp_rules: Dict[str, Any] = {}

    def instrument(self, envoy: Any) -> None:
        """Record what each side of this envoy's module is at the handoff.

        Called as the tree is built and again on dispatch (`Envoy._update`), which
        is when a module first carries transformers' ``_hf_tp_plan`` stamp.

        Raises:
            UnsupportedParallelStyle: for a style there is no rule for — refused
                up front rather than silently handing users a fragment.
        """
        super().instrument(envoy)

        module = envoy._module
        style = getattr(module, "_hf_tp_plan", None)
        if style is None:
            return

        if style not in SIDES:
            why = UNSUPPORTED.get(style, "not a parallel style this version of nnsight recognizes")
            raise UnsupportedParallelStyle(
                f"'{envoy.path}' is sharded as '{style}' ({why}), which interventions "
                "can't be shown whole, so this model can't be traced tensor-parallel."
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
        for side, kind in SIDES[style].items():
            self.tp_rules[f"{envoy.path}.{side}"] = (mesh, kind)

    def fragmented(self, location: str) -> bool:
        """Whether this location's value is one rank's slice.

        A dict lookup: the rules were recorded at instrument time, so nothing is
        inspected here and nothing branches on rank.
        """
        return location in self.tp_rules

    def whole(self, location: str, value: Any) -> Any:
        """The real tensor: every rank's slice gathered, or every rank's partial summed."""
        mesh, kind = self.tp_rules[location]
        return _reduce(value, mesh) if kind == "partial" else _gather(value, mesh)

    def fragment(self, location: str, whole: Any) -> Any:
        """What this rank hands its forward back, carrying whatever the workers left.

        A shard's slice again; for a partial, the whole on rank 0 and zeros on the
        others, so the module's own post-hook reduce yields the whole everywhere.
        """
        mesh, kind = self.tp_rules[location]
        if kind == "partial":
            keep = mesh.get_local_rank() == 0
            return apply(
                whole,
                lambda tensor: tensor if keep or not tensor.is_floating_point() else torch.zeros_like(tensor),
                torch.Tensor,
            )
        return _reshard(whole, mesh)

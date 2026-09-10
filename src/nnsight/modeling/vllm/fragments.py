"""Which values a vLLM engine splits across ranks, and how to reassemble them.

vLLM shards its linear layers under tensor parallelism, so the value at a
``ColumnParallelLinear`` or ``RowParallelLinear`` is one rank's piece of the real
tensor. A user asked for the layer, not a piece of it, so those are gathered
before a worker sees them and re-split before vLLM's own forward carries on.

A *fused* column-parallel layer takes one more step. ``QKVParallelLinear`` and
``MergedColumnParallelLinear`` pack several projections into one weight and each
rank holds a slice of every one of them, so the ranks' concatenation comes out
``[q0 k0 v0 | q1 k1 v1]`` rather than ``[q | k | v]``. The gather un-interleaves
that back into the layout a single rank has and the write-back re-interleaves it,
because per-head slicing and ``gate, up = value.chunk(2, -1)`` are what these
values are read for and both are silently wrong on the rank-major order.

A ``FusedMoE`` layer needs the same correction for a different reason: an MoE
block that defers the combine (``reduce_results=False`` — Qwen-MoE, DeepSeek)
returns per-rank partial sums that the *outer block* all-reduces afterwards, so
the value at the experts module is a partial too. It is gathered and re-split on
the same terms, with the group size taken from the expert layout rather than
``tp_size`` alone.

Everything about *when* — once per visit, only when something is waiting, put
back on the way out — belongs to
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle] and is
shared with every other distributed runtime; see
[`nnsight.intervention.fragments`][nnsight.intervention.fragments].

That sharing is what this module is. Doing the same job from
[`VLLMBatcher`][nnsight.modeling.vllm.batching.VLLMBatcher] instead costs two
extra pairs of forward hooks per parallel layer, bracketing the interleaver's
own, plus a memo and an explicit release: ``Batcher.narrow`` runs once per
*parked worker*, so several workers reading one value would otherwise run several
collectives and deadlock the ranks. On the interleaver the bracket is already
once-per-visit, so none of that is needed — no memo, no ``watch``/``release``, no
extra hooks.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Dict, List, Tuple

import torch

from ...intervention.fragments import Fragments
from ...util import apply


def _tp_world_size() -> int:
    """How many ranks this engine's tensor-parallel group spans.

    The one caller is `VLLMFragments.instrument`, which runs while a worker builds
    its tree in `NNsightGPUModelRunner.load_model` — after vLLM has initialized the
    group. The fallback is for a tree built anywhere else: ``1`` is also the answer
    that means "nothing here is a fragment", so degrading to it leaves an
    unsharded tree rather than failing the load.
    """
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )

        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


class VLLMFragments(Fragments):
    """Records which of an engine's locations hold one rank's piece.

    Attributes:
        enabled: Whether this engine is sharded at all. False on one rank, and
            then this costs one attribute check per handled location.
        rules: Location -> the module that produced it and which side it is.
            The module is kept because the collective to run and the arithmetic
            to undo it both depend on its type and its ``tp_size``/``tp_rank``.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.rules: Dict[str, Tuple[torch.nn.Module, str]] = {}

    def instrument(self, envoy: Any) -> None:
        """Record whether either side of this envoy's module is a piece.

        Called for every envoy as the tree is built, which is the one moment both
        the module and its path are in hand. Holding onto the module is safe for
        the same reason the previous hook-based version was: vLLM builds the tree
        once, after ``load_model``, and does not swap modules under it — a hook
        registered on a swapped-out module would have gone just as dead.
        """
        if _tp_world_size() < 2:
            return

        module = envoy._module
        for side in ("input", "output"):
            if _is_piece(module, side):
                self.rules[f"{envoy.path}.{side}"] = (module, side)
                self.enabled = True

    def fragmented(self, location: str) -> bool:
        return location in self.rules

    def whole(self, location: str, value: Any) -> "tuple[Any, Any]":
        """The real tensor behind ``value``, and how to cut it back down.

        The whole is in the layout a single rank would have produced, fused
        projections included, so a recipe written against ``tp=1`` reads and
        writes the same features here.

        vLLM's rules describe a location and nothing else — no value here carries
        its own layout — so the way back is `split` with this location bound to it.
        """
        from vllm.distributed.communication_op import (
            tensor_model_parallel_all_gather,
            tensor_model_parallel_all_reduce,
        )
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        module, side = self.rules[location]

        if isinstance(module, ColumnParallelLinear) or side == "input":
            # Column sharding splits the output features, so the ranks hold
            # different columns of the same rows; a row-parallel layer takes its
            # input already split by feature. Either way the whole is the ranks'
            # concatenation.
            collective = tensor_model_parallel_all_gather
            # A layer sharded across decode-context-parallel *groups* (vLLM's
            # DCPGroupColumnParallelLinear) is replicated within each group, so
            # the ranks' concatenation carries every shard `group_size` times.
            groups = getattr(module, "group_size", 1)
            if groups > 1:
                world = groups * module.tp_size
                collective = lambda tensor: _one_per_group(tensor_model_parallel_all_gather(tensor), world, groups)
            # A fused layer's own piece is `[q_r | k_r | v_r]`, so the ranks'
            # concatenation interleaves the projections. Undoing it here is what
            # makes the promise the whole gather is for: the same slice means the
            # same feature at every `tp_size`.
            layout = _fused_sub_shards(module) if isinstance(module, ColumnParallelLinear) else None
            if layout is not None:
                gather, (widths, replicas) = collective, layout
                collective = lambda tensor: _unfuse(gather(tensor), widths, replicas, module.tp_size)
        else:
            # Row sharding splits the summed terms, and a deferred-combine FusedMoE
            # leaves each rank a partial sum of the experts' output — either way
            # each rank holds part of the total and the whole is their sum. For the
            # MoE it is the same collective the outer block runs right afterwards.
            collective = tensor_model_parallel_all_reduce

        return apply(value, collective, torch.Tensor), partial(self.split, location)

    def split(self, location: str, whole: Any) -> Any:
        """This rank's piece of ``whole``, as vLLM's own forward expects it.

        Applied to whatever intervention code left behind, so an edit made to the
        assembled tensor is carried back into the model rather than dropped — and
        to a value that was never gathered at all (a `.skip` replacement, or the
        argument of an ad-hoc call), which is already the real tensor. Either way
        the value is in the single-rank layout, so a fused layer's piece is
        re-interleaved as well as narrowed.
        """
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            split_tensor_along_last_dim,
        )

        module, side = self.rules[location]
        moe = _moe_layer()

        if moe is not None and isinstance(module, moe):
            # The block all-reduces this right after the module returns, so hand
            # back an equal share rather than the whole: dividing by the
            # collective's group size gives partials the block's own reduce sums
            # back to the value exactly once instead of double-counting it.
            group_size = _moe_group_size(module)
            return apply(whole, lambda tensor: tensor / group_size, torch.Tensor)

        if isinstance(module, ColumnParallelLinear) or side == "input":
            layout = _fused_sub_shards(module) if isinstance(module, ColumnParallelLinear) else None
            if layout is not None:
                widths, replicas = layout
                return apply(
                    whole,
                    lambda tensor: _fuse(
                        tensor, widths, replicas, module.tp_size, module.tp_rank
                    ),
                    torch.Tensor,
                )
            return apply(
                whole,
                lambda tensor: split_tensor_along_last_dim(
                    tensor, num_partitions=module.tp_size
                )[module.tp_rank].contiguous(),
                torch.Tensor,
            )
        # An all-reduce summed every rank's partial; dividing evenly gives a set of
        # partials that sums back to it.
        return apply(whole, lambda tensor: tensor / module.tp_size, torch.Tensor)


def _one_per_group(gathered: torch.Tensor, world: int, group_size: int) -> torch.Tensor:
    """Drop the replicas from an all-gather over ranks that hold their shard in groups.

    ``gathered`` is ``world`` shards along the last dim, rank order; ranks
    ``g*group_size .. (g+1)*group_size-1`` hold the same shard, so one per group is
    the whole.
    """
    chunks = gathered.chunk(world, dim=-1)
    return torch.cat(chunks[::group_size], dim=-1)


def _fused_sub_shards(module: Any) -> "tuple[List[int], List[int]] | None":
    """The projections packed into one rank's piece of a column-parallel layer.

    Their widths *on one rank*, in packing order, and how many adjacent ranks
    hold a copy of each. ``None`` for a layer whose piece is a single block,
    which is every column-parallel layer but the two fused ones and needs no
    reordering at all.

    The widths are vLLM's own: ``ColumnParallelLinear.__init__`` divides
    ``output_sizes`` by ``tp_size`` into ``output_partition_sizes`` for exactly
    the two subclasses that set ``output_sizes``, and leaves one entry — the
    whole shard — for everything else. Read against vLLM 0.27.1.

    Replication is only ever read off ``num_kv_head_replicas``, so a merged
    column that replicates a sub-shard some other way is not covered and does
    not warn: ``_KimiGDNMergedColumnParallelLinear`` gives every rank the same
    copy of one projection (``output_sizes[i] *= tp_size``, loaded with
    ``tp_rank`` forced to 0), which reads here as an ordinary merged column and
    comes back ``tp_size`` copies of it too wide.
    """
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )

    widths = list(getattr(module, "output_partition_sizes", ()))
    if len(widths) < 2:
        return None

    if isinstance(module, QKVParallelLinear) and len(widths) == 3:
        # Q is sharded across every rank, but a model with fewer KV heads than
        # ranks has its K and V replicated instead: vLLM's weight loader takes
        # `shard_rank = tp_rank // num_kv_head_replicas`, so that many adjacent
        # ranks hold the same K and V and the gather carries each of them once
        # per rank in the group.
        replicas = getattr(module, "num_kv_head_replicas", 1)
        return widths, [1, replicas, replicas]

    if isinstance(module, MergedColumnParallelLinear):
        return widths, [1] * len(widths)

    # Some other subclass packs projections nnsight has no layout for — a QKV
    # with an indexer (MiniMax-M3) is one. Leaving it in rank order is the only
    # safe answer, but a silent one would read exactly like the layout this
    # module exists to remove.
    warnings.warn(
        f"{type(module).__name__} packs {len(widths)} projections into each rank's"
        " shard and nnsight has no layout for it, so its gathered value stays in"
        " rank order ([q0 k0 v0 | q1 k1 v1], not [q | k | v]). Slice it per rank."
    )
    return None


def _unfuse(
    gathered: torch.Tensor, widths: List[int], replicas: List[int], tp_size: int
) -> torch.Tensor:
    """Rank-major to projection-major: ``[q0 k0 v0 | q1 k1 v1]`` -> ``[q | k | v]``.

    The ranks sharing a replicated projection are contiguous, so one rank out of
    every ``replicas`` of them carries it exactly once — which is also what makes
    the result the single-rank width rather than a KV shard per rank.
    """
    pieces = gathered.chunk(tp_size, dim=-1)

    parts = []
    offset = 0
    for width, every in zip(widths, replicas):
        parts.extend(piece[..., offset : offset + width] for piece in pieces[::every])
        offset += width
    return torch.cat(parts, dim=-1)


def _fuse(
    whole: torch.Tensor,
    widths: List[int],
    replicas: List[int],
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """One rank's ``[q_r | k_r | v_r]`` back out of a whole ``[q | k | v]``.

    `_unfuse` for a single rank, which is all `split` ever needs: building the
    rank-major whole and then dropping every other rank's piece would cost a
    second full-width tensor to reach the same columns.
    """
    pieces = []
    offset = 0
    for width, every in zip(widths, replicas):
        start = offset + width * (tp_rank // every)
        pieces.append(whole[..., start : start + width])
        offset += width * (tp_size // every)
    return torch.cat(pieces, dim=-1)


def _moe_layer() -> Any:
    """The class a model's fused-experts module is, or None if there isn't one.

    ``FusedMoE`` through vLLM 0.26; from 0.27 the layer was rebuilt around a
    factory and a modular kernel, and the thing a model holds is a ``MoERunner``.
    Absent is not an error — it only means nothing in this tree can be a
    fused-experts module, which is the right answer for a vLLM that has neither.
    """
    from vllm.model_executor.layers import fused_moe

    for name in ("FusedMoE", "MoERunner"):
        found = getattr(fused_moe, name, None)
        if isinstance(found, type):
            return found
    return None


def _moe_group_size(module: Any) -> int:
    """How many ranks this MoE layer's experts are spread over.

    Expert parallelism reassigns the same ranks — the module-internal ``tp_size``
    drops to 1 and ``ep_size`` becomes the group — so the product is the group
    size under both layouts. From 0.27 these live on the layer's config rather
    than on the layer.
    """
    config = getattr(module, "moe_config", None)
    if config is not None:
        parallel = config.moe_parallel_config
        return parallel.tp_size * parallel.ep_size
    return module.tp_size * module.ep_size


def _is_piece(module: torch.nn.Module, side: str) -> bool:
    """Whether ``module``'s ``side`` really holds one rank's piece.

    Asked once per module at load rather than on every forward, so it may be as
    particular as it likes about the cases vLLM already handles itself.
    """
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    moe = _moe_layer()

    # A layer built with `disable_tp=True` is replicated on every rank rather than
    # sharded — vLLM sets its `tp_size` to 1 and guards its own collectives on
    # `tp_size > 1` for exactly that reason (DeepSeek-V2's `fused_qkv_a_proj` is
    # one; the branch beside it uses a `ReplicatedLinear` for the same role). The
    # engine's world size says nothing about it, so ask the module.
    if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
        if module.tp_size == 1:
            return False

    if isinstance(module, ColumnParallelLinear):
        # vLLM gathers this itself when asked to, and then it isn't a piece.
        return side == "output" and not module.gather_output

    if isinstance(module, RowParallelLinear):
        if side == "input":
            return module.input_is_parallel
        return not module.reduce_results

    if moe is not None and isinstance(module, moe):
        # Only the output is ever a piece: the inputs (hidden states, router
        # logits) are full replicated tensors under every expert layout.
        if side != "output" or _moe_group_size(module) <= 1:
            return False

        # Which vLLM's layer this is, asked by the flag that decides it rather
        # than by a `moe_config`, which both eras have.
        if hasattr(module, "reduce_results"):
            # Through 0.26 the layer carries its own flags. `reduce_results=True`
            # (Mixtral) reduces inside forward, and a combine kernel that already
            # reduced across ranks leaves nothing to gather — neither was ever
            # exposed as a partial.
            return not module.reduce_results and (
                not module.must_reduce_shared_expert_outputs()
            )

        config = getattr(module, "moe_config", None)
        if config is None:
            return False

        # From 0.27 the layer reduces its own output, and the conditions under
        # which it does not are the ones below — the negation of the guard in
        # `MoERunner._maybe_all_reduce`. Measured on a two-rank Qwen1.5-MoE: with
        # none of them set, both ranks hand back the identical tensor, so there is
        # nothing to gather.
        if getattr(module, "_fused_output_is_reduced", False):
            return False
        if getattr(config, "is_sequence_parallel", False):
            # Split by rows rather than left as a partial sum: a real fragment,
            # but one wanting concatenation rather than a sum, which nothing here
            # does yet. Leaving it alone reads one rank's rows; gathering it as a
            # partial would be arithmetic on unrelated tokens.
            return False
        return bool(getattr(config, "skip_final_all_reduce", False))

    return False

"""Which values a vLLM engine splits across ranks, and how to reassemble them.

vLLM shards its linear layers under tensor parallelism, so the value at a
``ColumnParallelLinear`` or ``RowParallelLinear`` is one rank's piece of the real
tensor. A user asked for the layer, not a piece of it, so those are gathered
before a worker sees them and re-split before vLLM's own forward carries on.

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

That sharing is what this module is. The same job used to be done by
[`VLLMBatcher`][nnsight.modeling.vllm.batching.VLLMBatcher] with two extra pairs
of forward hooks per parallel layer, installed by the model runner either side of
building the Envoy tree so they bracketed the interleaver's own. It needed them
because ``Batcher.narrow`` runs once per *parked worker*, so the gather had to be
memoized and explicitly released — several workers reading one value would
otherwise have run several collectives and deadlocked the ranks. Sitting on the
interleaver instead, the bracket is already once-per-visit and none of that is
needed: no memo, no ``watch``/``release``, no extra hooks.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

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

    def whole(self, location: str, value: Any) -> Any:
        """The real tensor behind ``value``, assembled from every rank's piece."""
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
        else:
            # Row sharding splits the summed terms, and a deferred-combine FusedMoE
            # leaves each rank a partial sum of the experts' output — either way
            # each rank holds part of the total and the whole is their sum. For the
            # MoE it is the same collective the outer block runs right afterwards.
            collective = tensor_model_parallel_all_reduce

        return apply(value, collective, torch.Tensor)

    def fragment(self, location: str, whole: Any) -> Any:
        """This rank's piece of ``whole``, as vLLM's own forward expects it.

        Applied to whatever intervention code left behind, so an edit made to the
        assembled tensor is carried back into the model rather than dropped.
        """
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            split_tensor_along_last_dim,
        )

        module, side = self.rules[location]

        if isinstance(module, FusedMoE):
            # The block all-reduces this right after the module returns, so hand
            # back an equal share rather than the whole: dividing by the
            # collective's group size gives partials the block's own reduce sums
            # back to the value exactly once instead of double-counting it.
            # Expert parallelism reassigns the same ranks (the module-internal
            # tp_size becomes 1 and ep_size becomes the group), so the product is
            # the group size under both layouts.
            group_size = module.tp_size * module.ep_size
            return apply(whole, lambda tensor: tensor / group_size, torch.Tensor)

        if isinstance(module, ColumnParallelLinear) or side == "input":
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


def _is_piece(module: torch.nn.Module, side: str) -> bool:
    """Whether ``module``'s ``side`` really holds one rank's piece.

    Asked once per module at load rather than on every forward, so it may be as
    particular as it likes about the cases vLLM already handles itself.
    """
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

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

    if isinstance(module, FusedMoE):
        # Only the output is ever a piece: the inputs (hidden states, router
        # logits) are full replicated tensors under both expert layouts.
        # `reduce_results=True` (Mixtral) reduces inside forward, and a combine
        # kernel that already reduced across ranks
        # (must_reduce_shared_expert_outputs) leaves nothing to gather — neither
        # was ever exposed as a partial.
        return (
            side == "output"
            and not module.reduce_results
            and module.tp_size * module.ep_size > 1
            and not module.must_reduce_shared_expert_outputs()
        )

    return False

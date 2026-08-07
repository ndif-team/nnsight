"""Scope a worker to its own request's tokens inside a scheduled step.

A stacked-tensor model gives each invoke a row range that is fixed the moment the
trace is written. vLLM gives neither. It packs every request the scheduler picked
for a step into one flat ``[total_tokens, hidden]`` slab — a whole prompt's tokens
on prefill, a single token per decode step — and which requests are in that slab
changes from step to step as they arrive and finish.

So a worker's group is a *token span*, recomputed every step by
[`NNsightGPUModelRunner`][nnsight.modeling.vllm.model_runners.GPUModelRunner.NNsightGPUModelRunner]
rather than assigned once up front. The row math itself is unchanged from
[`Batcher`][nnsight.intervention.batching.Batcher]: dim 0 is the token axis, so
narrowing to ``[start, size]`` selects exactly a request's tokens.

Under tensor parallelism a second correction is needed. vLLM splits its linear
layers across ranks, so the value at a ``ColumnParallelLinear`` or
``RowParallelLinear`` is one rank's shard of the real tensor. A user asked for the
layer, not a shard of it, so those values are gathered before a worker sees them
and re-split before vLLM's own forward carries on.

A ``FusedMoE`` layer needs the same correction for a different reason: an MoE block
that defers the combine (``reduce_results=False`` — Qwen-MoE, DeepSeek) returns
per-rank partial sums that the *outer block* all-reduces afterwards, so the value at
the experts module is a partial too. It is gathered and re-split on the same terms,
with the group size taken from the expert layout rather than ``tp_size`` alone.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ...intervention.batching import Batcher
from ...util import apply


class VLLMBatcher(Batcher):
    """A [`Batcher`][nnsight.intervention.batching.Batcher] over vLLM's flat token axis.

    Attributes:
        module: The parallel linear currently running, or None outside one.
        type: Whether ``module``'s input or output is being served.
        parallel: Whether ``module``'s value is really a shard.
        gathered: The whole tensor, once assembled from every rank's shard, or None.
    """

    def __init__(self, envoy: Optional[object] = None) -> None:
        super().__init__(envoy)
        self.module: Any = None
        self.type: Optional[str] = None
        self.parallel: bool = False
        self.gathered: Any = None

    @property
    def batching(self) -> bool:
        """Whether narrowing applies — always.

        The base skips narrowing for a lone invoke, because one invoke *is* the
        whole batch. That never holds here: the engine fills a step with whatever it
        has, so a request's tokens sit alongside other requests' — another trace's,
        another tenant's, or a decode of a request whose own block already finished.
        A worker is only ever entitled to its own span, so there is no case in which
        handing it the whole slab is right.
        """
        return True

    def narrow(self, value: Any, group: Any) -> Any:
        return super().narrow(self._whole(value), group)

    def widen(self, full: Any, group: Any, edited: Any) -> Any:
        whole = super().widen(self._whole(full), group, edited)
        # An edit produces a fresh whole; keep it so a later worker in the same
        # module turn reads the edit, and so release re-shards the edited value
        # rather than the pre-edit gather.
        if self.parallel and self.gathered is not None:
            self.gathered = whole
        return whole

    def _whole(self, value: Any) -> Any:
        """The real tensor behind ``value``, assembled from every rank's shard.

        Only the parallel linears shard anything, so everywhere else this is the
        value itself. The result is kept for the rest of the module's turn: workers
        read and edit this one tensor in place, and [`release`][nnsight.modeling.vllm.batching.VLLMBatcher.release] reshards it back
        to the model afterwards. Keeping it also means each rank takes part in the
        gather collective once per value rather than once per worker — several
        workers reading the same value would otherwise deadlock the ranks.
        """
        if not self.parallel:
            return value
        if self.gathered is not None:
            return self.gathered

        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from vllm.distributed.communication_op import (
            tensor_model_parallel_all_gather,
            tensor_model_parallel_all_reduce,
        )

        if isinstance(self.module, ColumnParallelLinear):
            # Column sharding splits the output features, so the ranks hold
            # different columns of the same rows.
            collective = tensor_model_parallel_all_gather
        elif self.type == "input":
            # A row-parallel layer takes its input already split by feature.
            collective = tensor_model_parallel_all_gather
        elif isinstance(self.module, FusedMoE):
            # Each rank holds a partial sum of the experts' combined output —
            # the same collective the outer block runs right afterwards.
            collective = tensor_model_parallel_all_reduce
        else:
            # Row sharding splits the summed terms, so each rank holds a partial
            # sum and the whole is their total.
            collective = tensor_model_parallel_all_reduce

        self.gathered = apply(value, collective, torch.Tensor)
        return self.gathered

    def _shard(self, value: Any) -> Any:
        """This rank's piece of ``value``, as vLLM's own forward expects it."""
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            split_tensor_along_last_dim,
        )

        module = self.module

        if isinstance(module, FusedMoE):
            # The block all-reduces this right after the module returns, so hand
            # back an equal share rather than the whole: dividing by the
            # collective's group size gives partials the block's own reduce sums
            # back to the value exactly once instead of double-counting it.
            # Expert parallelism reassigns the same ranks (the module-internal
            # tp_size becomes 1 and ep_size becomes the group), so the product is
            # the group size under both layouts.
            group_size = module.tp_size * module.ep_size
            return apply(value, lambda tensor: tensor / group_size, torch.Tensor)

        if isinstance(module, ColumnParallelLinear) or self.type == "input":
            return apply(
                value,
                lambda tensor: split_tensor_along_last_dim(
                    tensor, num_partitions=module.tp_size
                )[module.tp_rank].contiguous(),
                torch.Tensor,
            )
        # An all-reduce summed every rank's partial; dividing evenly gives a set of
        # partials that sums back to it.
        return apply(value, lambda tensor: tensor / module.tp_size, torch.Tensor)

    def watch(self, module: torch.nn.Module, kind: str) -> None:
        """Note that ``module``'s ``kind`` is about to be served to workers."""
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        self.module = module
        self.type = kind
        self.gathered = None

        if isinstance(module, ColumnParallelLinear):
            # vLLM gathers this itself when asked to, and then it isn't a shard.
            self.parallel = not module.gather_output if kind == "output" else False
        elif isinstance(module, RowParallelLinear):
            if kind == "input":
                self.parallel = module.input_is_parallel
            else:
                self.parallel = not module.reduce_results
        elif isinstance(module, FusedMoE):
            # Only the output is ever a shard: the inputs (hidden states, router
            # logits) are full replicated tensors under both expert layouts.
            # `reduce_results=True` (Mixtral) reduces inside forward, and a combine
            # kernel that already reduced across ranks
            # (must_reduce_shared_expert_outputs) leaves nothing to gather —
            # neither was ever exposed as a partial.
            self.parallel = (
                kind == "output"
                and not module.reduce_results
                and module.tp_size * module.ep_size > 1
                and not module.must_reduce_shared_expert_outputs()
            )
        else:
            self.parallel = False

    def release(self, value: Any) -> Any:
        """Hand vLLM back what it should carry on with, and stop watching.

        If the value was gathered, the workers read and edited *that* whole tensor,
        so what vLLM continues from is a fresh shard of it — not the shard it
        produced, which the workers never saw. An edit rebuilds the whole through
        `widen`, which writes it back onto ``gathered``, so re-sharding
        ``gathered`` carries the edit; a read leaves ``gathered`` as the plain
        assembly. When nothing gathered (the common, unsharded case) the value
        passes straight through.
        """
        if self.parallel and self.gathered is not None:
            value = self._shard(self.gathered)
        self.module = None
        self.type = None
        self.parallel = False
        self.gathered = None
        return value

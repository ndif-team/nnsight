"""Envoys for modules vLLM split across ranks.

Interleaving already makes a sharded *activation* whole: the value at a location
is gathered on the way to a worker and re-split on the way back into vLLM's own
forward, once per visit, by
[`VLLMFragments`][nnsight.modeling.vllm.fragments.VLLMFragments]. That covers
everything read at a location — ``.output``, ``.input`` — because what drives it
is the model firing its own hooks, and every rank fires them alike.

What it does not cover is an **ad-hoc call**: a logit lens runs ``lm_head`` on
an intermediate hidden state, away from that module's place in the forward pass.
The caller is holding, and wants back, whole tensors, but a parallel layer's
forward expects this rank's piece and returns this rank's piece — so the input is
cut down on the way in and the output reassembled on the way out, off the rules
[`VLLMFragments`][nnsight.modeling.vllm.fragments.VLLMFragments] already recorded
for exactly this envoy's two locations.

``layer.weight`` is this rank's real slice here, as it is anywhere else in vLLM.
``layer.param("weight")`` is the whole: the ranks' slices gathered by the parameter's
own sharding metadata, with a merged projection's components and a
vocab-parallel table's padding put back in order.
"""

from __future__ import annotations

from typing import Any

import torch

from ...intervention.envoy import Envoy


def parallel_envoys() -> dict:
    """The ``envoys`` map pairing vLLM's parallel layers with `ParallelEnvoy`.

    Keys are matched against a module's MRO, so the merged subclasses
    (``QKVParallelLinear``, ``MergedColumnParallelLinear``, ``ParallelLMHead``)
    are covered by their bases. Built on demand rather than at import so the
    module stays importable without vLLM.
    """
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    return {
        ColumnParallelLinear: ParallelEnvoy,
        RowParallelLinear: ParallelEnvoy,
        VocabParallelEmbedding: ParallelEnvoy,
    }


def _whole_parameter(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """``tensor``, a parameter or buffer of ``module``, gathered across the
    tensor-parallel group into the whole it is a slice of.

    vLLM stamps a sharded parameter with the axes it could be split along; which
    one the ranks did split follows the layer: a row-parallel layer splits its
    ``input_dim``, every other parallel layer its ``output_dim``. A parameter
    without that stamp is replicated and returned as is.
    A merged column-parallel projection holds its components stacked per rank
    (``[gate_r; up_r]``), so the gathered rows are regrouped per component. A
    QKV projection is the same with q, k, v, where a k or v head replicated
    across ranks is kept once. A vocab-parallel table is padded per rank and
    reindexed into token order by the module's own mapping.
    """
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from .fragments import _tp_world_size

    world = _tp_world_size()
    if world == 1:
        return tensor

    if isinstance(module, VocabParallelEmbedding):
        gathered = tensor_model_parallel_all_gather(tensor, dim=0)
        order = module.get_sharded_to_full_mapping()[: module.num_embeddings]
        return gathered[torch.tensor(order, device=gathered.device)]

    # Presence is the signal: vLLM stamps only parameters it shards.
    axis = "input_dim" if isinstance(module, RowParallelLinear) else "output_dim"
    dim = getattr(tensor, axis, None)
    if dim is None:
        return tensor
    gathered = tensor_model_parallel_all_gather(tensor, dim=dim)

    if isinstance(module, QKVParallelLinear):
        head, v_head = module.head_size, module.v_head_size
        sizes = [module.num_heads * head, module.num_kv_heads * head, module.num_kv_heads * v_head]
        keep_every = [1, module.num_kv_head_replicas, module.num_kv_head_replicas]
    elif isinstance(module, MergedColumnParallelLinear):
        sizes = [size // world for size in module.output_sizes]
        keep_every = [1] * len(sizes)
    else:
        return gathered

    if sum(sizes) != tensor.shape[dim]:
        return gathered
    shards = gathered.split(tensor.shape[dim], dim=dim)
    components = [shard.split(sizes, dim=dim) for shard in shards]
    return torch.cat(
        [
            torch.cat([components[rank][i] for rank in range(0, world, every)], dim=dim)
            for i, every in enumerate(keep_every)
        ],
        dim=dim,
    )


class ParallelEnvoy(Envoy):
    """An envoy over a module holding one rank's piece of a larger one.

    Behaves exactly as an [`Envoy`][nnsight.intervention.envoy.Envoy] on an
    unsharded engine — the corrections below are keyed off
    [`VLLMFragments`][nnsight.modeling.vllm.fragments.VLLMFragments], which finds
    nothing to do on one rank.
    """

    def __call__(self, *args: Any, hook: bool = False, **kwargs: Any) -> Any:
        """Run this module's forward ad hoc, on whole tensors either side.

        A parallel layer's forward expects this rank's piece and returns this
        rank's piece, but a caller reaching for the module ad hoc is holding, and
        wants back, the real thing. So the input is cut down to this rank's share
        on the way in and the output reassembled on the way out — which is what
        [`VLLMFragments`][nnsight.modeling.vllm.fragments.VLLMFragments] already
        knows how to do, keyed by exactly the two locations `instrument` recorded
        for this envoy.

        Every rank runs the block, so every rank reaches the same collectives in
        the same order — as long as the call itself is not under rank-dependent
        control flow, which is the same condition every other collective here
        carries.
        """
        fragments = self.interleaver.fragments

        if fragments is None or not fragments.enabled:
            return super().__call__(*args, hook=hook, **kwargs)

        into, outof = f"{self.path}.input", f"{self.path}.output"

        if fragments.fragmented(into):
            # `split`, not the way back from a gather: the caller's tensor is
            # whole because they are holding it, not because anything assembled
            # it — and this call may be nested inside that location's own open
            # handoff, whose record must stay untouched.
            args, kwargs = fragments.split(into, (args, kwargs))

        result = super().__call__(*args, hook=hook, **kwargs)

        if fragments.fragmented(outof):
            # `whole` hands back ``(whole, undo)``. The undo exists to carry an
            # edit back into the model's own forward; an ad-hoc call has no
            # forward to return to — the whole *is* the return value — so it is
            # dropped rather than applied.
            result, _ = fragments.whole(outof, result)

        return result

    def _parameter(self, name: str) -> torch.Tensor:
        """The whole parameter: this rank's slice gathered across the group."""
        return _whole_parameter(self._module, super()._parameter(name))

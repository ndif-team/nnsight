"""Scope a worker to its own request's tokens inside a scheduled step.

A stacked-tensor model gives each invoke a row range that is fixed the moment the
trace is written. vLLM gives neither. It packs every request the scheduler picked
for a step into one flat ``[total_tokens, hidden]`` slab — a whole prompt's tokens
on prefill, a single token per decode step — and which requests are in that slab
changes from step to step as they arrive and finish.

So a worker's group is a *token span*, recomputed every step by
[`NNsightGPUModelRunner`][nnsight.modeling.vllm.model_runners.GPUModelRunner.NNsightGPUModelRunner]
rather than assigned once up front. The row math itself is the base
[`Batcher`][nnsight.intervention.batching.Batcher]'s, moved off dim 0 onto
whichever axis carries tokens: dim 0 for a model vLLM has its own definition
for, dim 1 for one served through vLLM's Transformers backend (see
`VLLMBatcher._token_dim`).

Gathering a sharded value is *not* here — see
[`fragments`][nnsight.modeling.vllm.fragments]. The split is the point: narrowing
happens once per parked worker, while a collective must happen once per value
however many workers read it.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...intervention.batching import Batcher


class VLLMBatcher(Batcher):
    """A [`Batcher`][nnsight.intervention.batching.Batcher] over vLLM's flat token axis."""

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

    def _token_dim(self, tensor: torch.Tensor) -> Optional[int]:
        """The axis of ``tensor`` the token spans index, or ``None`` if it has none.

        A model vLLM has its own definition for emits 2-D activations
        ``[total_tokens, hidden]``, so tokens are dim 0 — the base layout.

        A model without one is served through vLLM's Transformers backend, which
        runs the HuggingFace module with a leading singleton batch dim
        (``inputs_embeds[None, ...]``). Its decoder layers emit
        ``[1, total_tokens, hidden]``, so tokens are dim 1. Reading dim 0 there
        finds ``1 != total``, calls the activation unbatched, and hands the block
        every request's tokens while dropping its writes — silently, since a
        passthrough looks exactly like a tensor that legitimately isn't batched.

        Anything matching neither really isn't batched (a per-layer constant, a
        weight) and passes through.
        """
        if tensor.shape[0] == self.total:
            return 0
        if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == self.total:
            return 1
        return None

    def _narrow_tensor(self, tensor: torch.Tensor, group: list) -> torch.Tensor:
        """Slice one slab down to ``group``'s token span.

        A view, so an in-place edit by the block lands in the slab the model goes
        on to read. The base marks its view ``_nnsight_batch`` for the backward
        pass to redirect through; nothing does that here, because backward isn't
        supported on the vLLM path at all.
        """
        dim = self._token_dim(tensor)
        if dim is None:
            return tensor
        start, size = group
        return tensor.narrow(dim, start, size)

    def _widen_tensor(
        self, full: torch.Tensor, group: list, edited: torch.Tensor
    ) -> torch.Tensor:
        """Splice ``edited`` back over ``group``'s span of ``full``."""
        dim = self._token_dim(full)
        if dim is None:
            return full
        start, size = group
        # cat rather than an in-place write: `edited` is often a narrowed view of
        # `full`, and writing a tensor into a slice of itself aliases.
        pre = full.narrow(dim, 0, start)
        post = full.narrow(dim, start + size, self.total - start - size)
        return torch.cat([pre, edited, post], dim=dim)

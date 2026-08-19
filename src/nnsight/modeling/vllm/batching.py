"""Scope a worker to its own request's tokens inside a scheduled step.

A stacked-tensor model gives each invoke a row range that is fixed the moment the
trace is written. vLLM gives neither. It packs every request the scheduler picked
for a step into one flat ``[total_tokens, hidden]`` slab — a whole prompt's tokens
on prefill, a single token per decode step — and which requests are in that slab
changes from step to step as they arrive and finish.

So a worker's group is a *token span*, recomputed every step by
[`NNsightGPUModelRunner`][nnsight.modeling.vllm.model_runners.GPUModelRunner.NNsightGPUModelRunner]
rather than assigned once up front. The row math itself is unchanged from
[`Batcher`][nnsight.intervention.batching.Batcher]: narrowing to ``[start,
size]`` along the token axis selects exactly a request's tokens. For a native
vLLM model that axis is dim 0; a model served through vLLM's Transformers
backend carries a leading singleton batch dim, so its token axis is dim 1 (see
`VLLMBatcher._batch_dim`).

Gathering a sharded value is *not* here — see
[`fragments`][nnsight.modeling.vllm.fragments]. It used to be, and the split is
the point: narrowing happens once per parked worker, while a collective must
happen once per value however many workers read it.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...intervention.batching import Batcher


class VLLMBatcher(Batcher):
    """A [`Batcher`][nnsight.intervention.batching.Batcher] over vLLM's flat token axis."""

    def __init__(self, envoy: Optional[object] = None) -> None:
        # The envoy is optional here alone: the model runner constructs this
        # before there is a tree to hand it, and nothing in the row math needs
        # one — the spans come from the scheduler, not from an invoke.
        super().__init__(envoy)

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

    def _batch_dim(self, tensor: torch.Tensor) -> Optional[int]:
        """Locate the token axis, accounting for the Transformers backend.

        A native vLLM model emits 2-D activations ``[total_tokens, hidden]``:
        the token axis is dim 0, which the base rule handles.

        A model without a native vLLM definition runs through vLLM's
        Transformers backend, which wraps the HuggingFace model and adds a
        leading singleton batch dim (``inputs_embeds[None, ...]``). Its
        decoder-layer activations are ``[1, total_tokens, hidden]``, so the
        token axis is dim 1. Without this, the base rule sees ``shape[0] == 1
        != total``, calls the tensor unbatched, and every read returns all
        requests' tokens while every write is silently dropped.
        """
        dim = super()._batch_dim(tensor)
        if dim is not None:
            return dim
        if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == self.total:
            return 1
        return None

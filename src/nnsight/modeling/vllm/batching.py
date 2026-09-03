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

Gathering a sharded value is *not* here — see
[`fragments`][nnsight.modeling.vllm.fragments]. The split is the point: narrowing
happens once per parked worker, while a collective must happen once per value
however many workers read it.
"""

from __future__ import annotations

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

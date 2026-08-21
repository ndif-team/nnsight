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

Parameters are left alone. ``layer.weight`` is this rank's real slice here, as it
is anywhere else in vLLM, and gathering one is the caller's business.
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
            args, kwargs = fragments.fragment(into, (args, kwargs))

        result = super().__call__(*args, hook=hook, **kwargs)

        if fragments.fragmented(outof):
            result = fragments.whole(outof, result)

        return result

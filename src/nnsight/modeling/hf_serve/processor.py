"""NNsight-aware ContinuousBatchProcessor.

Subclasses HF's ``ContinuousBatchProcessor`` to inject nnsight
interleaving around the model forward pass and sampling, mirroring
``NNsightGPUModelRunner`` for vLLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from transformers.generation.continuous_batching.continuous_api import ContinuousBatchProcessor
from transformers.generation.logits_process import LogitsProcessorList

if TYPE_CHECKING:
    from ..language import LanguageModel
    from ..common.request_helper import NNsightRequestHelper


class NNsightCBProcessor(ContinuousBatchProcessor):
    """ContinuousBatchProcessor with nnsight intervention hooks.

    Overrides ``_model_forward`` and ``_sample`` to wrap them in the
    interleaver context so nnsight hooks fire on module inputs/outputs
    during the forward pass.
    """

    def __init__(
        self,
        *args,
        nnsight_model: "LanguageModel" = None,
        request_helper: "NNsightRequestHelper" = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nnsight_model = nnsight_model
        self.request_helper = request_helper

    def _model_forward(self, model: nn.Module, batch_data: dict) -> torch.Tensor:
        """Forward pass wrapped with nnsight interleaving."""
        nns = self.nnsight_model
        if nns is None or not nns.interleaver.mediators:
            return super()._model_forward(model, batch_data)

        interleaver = nns.interleaver
        # The HF paged manager reads ``mediator.deferred_exception``
        # itself when finalizing requests; raising from ``__exit__``
        # in the bg generation thread would crash the engine.
        interleaver.defer_exceptions = True

        try:
            with interleaver:
                logits = super()._model_forward(model, batch_data)

                # Hook logits for user access via model.logits.output
                logits = nns.logits(logits, hook=True)

                # Unflatten: remap batch groups from token-level to
                # request-level for sampling hooks
                self.request_helper.unflatten(nns)

        finally:
            interleaver.defer_exceptions = False

        return logits

    def _sample(
        self,
        probs: torch.Tensor,
        logits_indices: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> None:
        """Sampling wrapped with nnsight interleaving."""
        nns = self.nnsight_model
        if nns is None or not nns.interleaver.mediators:
            return super()._sample(probs, logits_indices, output_ids)

        interleaver = nns.interleaver
        # Same reasoning as ``_model_forward`` — engine reads
        # ``mediator.deferred_exception`` itself; raising from
        # ``__exit__`` here would crash the engine.
        interleaver.defer_exceptions = True

        try:
            with interleaver:
                super()._sample(probs, logits_indices, output_ids)

                # Hook sampled tokens for user access
                nns.samples(output_ids, hook=True)

        finally:
            interleaver.defer_exceptions = False

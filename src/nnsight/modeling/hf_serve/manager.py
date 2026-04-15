"""NNsight-aware ContinuousBatchingManager.

Subclasses HF's ``ContinuousBatchingManager`` to:
1. Create an ``NNsightCBProcessor`` instead of the base processor
2. Assign mediator batch groups each step from scheduler output
3. Collect saves and finalize mediators when requests finish

Mirrors ``NNsightLLMEngine`` + ``NNsightGPUModelRunner._update_states``
from the vLLM integration.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from transformers.generation.continuous_batching.continuous_api import (
    ContinuousBatchingManager,
    ContinuousBatchProcessor,
)
from transformers.generation.configuration_utils import (
    ContinuousBatchingConfig,
    GenerationConfig,
)

from .processor import NNsightCBProcessor
from .batching import HFBatcher

if TYPE_CHECKING:
    from ..language import LanguageModel
    from ..common.request_helper import NNsightRequestHelper


class NNsightCBManager(ContinuousBatchingManager):
    """ContinuousBatchingManager with nnsight intervention support.

    Wraps the generation loop to assign mediator batch groups per step
    and collect saved values when requests finish.

    Args:
        model: The raw HF ``PreTrainedModel``.
        generation_config: HF generation config.
        nnsight_model: A ``LanguageModel`` wrapping the same model.
        request_helper: Shared mediator lifecycle helper.
    """

    def __init__(
        self,
        model: Any,
        generation_config: GenerationConfig,
        nnsight_model: "LanguageModel" = None,
        request_helper: "NNsightRequestHelper" = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        **kwargs,
    ):
        if continuous_batching_config is None:
            continuous_batching_config = ContinuousBatchingConfig()

        super().__init__(
            model=model,
            generation_config=generation_config,
            continuous_batching_config=continuous_batching_config,
            **kwargs,
        )

        self.nnsight_model = nnsight_model
        self.request_helper = request_helper

    def _create_batch_processor(self) -> ContinuousBatchProcessor:
        """Override to create NNsightCBProcessor instead of base."""
        from transformers.generation.continuous_batching.cache import PagedAttentionCache
        from transformers.generation.continuous_batching.scheduler import (
            SCHEDULER_MAPPING,
            FIFOScheduler,
        )
        from transformers.generation.continuous_batching.requests import logger

        # Create the PagedAttentionCache
        paged_attention_cache = PagedAttentionCache(
            self.model.config,
            self.continuous_batching_config,
            self.model.device,
            self.model.dtype,
            tp_size=getattr(self.model, "_tp_size", None),
        )
        self._use_prefix_sharing = paged_attention_cache.use_prefix_sharing

        # Create the scheduler
        scheduler_type = self.continuous_batching_config.scheduler_type
        scheduler = SCHEDULER_MAPPING.get(scheduler_type, None)
        if scheduler is None:
            logger.warning(f"Scheduler '{scheduler_type}' not found. Defaulting to FIFO.")
            scheduler = FIFOScheduler

        # Create the NNsight-aware batch processor
        processor = NNsightCBProcessor(
            cache=paged_attention_cache,
            config=self.model.config,
            generation_config=self.generation_config,
            continuous_batching_config=self.continuous_batching_config,
            input_queue=self.input_queue,
            output_router=self.output_router,
            stop_event=self.stop_event,
            model_device=self.model.device,
            model_dtype=self.model.dtype,
            scheduler=scheduler(paged_attention_cache),
            nnsight_model=self.nnsight_model,
            request_helper=self.request_helper,
        )

        # Set up the batcher on the nnsight model's interleaver
        if self.nnsight_model is not None:
            self.nnsight_model._interleaver.batcher = HFBatcher(batch_dim=1)
            self.nnsight_model._interleaver.mediators = []

        return processor

    def _inner_generation_loop(self, batch_processor: ContinuousBatchProcessor) -> None:
        """Override to assign mediator batch groups and collect saves."""
        if not batch_processor.prepare_next_batch():
            self._has_new_requests.wait(timeout=0.1)
            self._has_new_requests.clear()
            return

        self._assign_batch_groups(batch_processor)
        self._generation_step()
        batch_processor.update_batch()
        self._collect_finished(batch_processor)

    def _assign_batch_groups(self, batch_processor: ContinuousBatchProcessor) -> None:
        """Assign ``mediator.batch_group`` from the scheduled batch."""
        if self.nnsight_model is None or self.request_helper is None:
            return

        requests_in_batch = batch_processor.inputs_and_outputs.requests_in_batch
        if not requests_in_batch:
            return

        num_tokens_scheduled = {}
        batch_req_ids = []

        cu_seq_lens = batch_processor.inputs_and_outputs.cumulative_seqlens_q
        for i, future_state in enumerate(requests_in_batch):
            req_id = future_state.state.request_id
            num_tokens = int(cu_seq_lens[i + 1].item()) - int(cu_seq_lens[i].item())
            num_tokens_scheduled[req_id] = num_tokens
            batch_req_ids.append(req_id)

        self.request_helper._batch_req_ids = batch_req_ids
        self.request_helper._num_scheduled_tokens = num_tokens_scheduled

        self.request_helper.process_batch_groups(
            num_tokens_scheduled, batch_req_ids, self.nnsight_model
        )

        self.nnsight_model._interleaver.batcher.needs_batching = (
            len(self.nnsight_model._interleaver.mediators) > 1
        )

    def _collect_finished(self, batch_processor: ContinuousBatchProcessor) -> None:
        """Collect saves from finished requests."""
        if self.nnsight_model is None or self.request_helper is None:
            return

        finished_ids = set()
        requests_in_batch = getattr(
            batch_processor.inputs_and_outputs, "requests_in_batch", []
        )
        for future_state in requests_in_batch:
            state = future_state.state
            if state.status is not None and state.status.name == "FINISHED":
                finished_ids.add(state.request_id)

        if not finished_ids:
            return

        matched = self.request_helper.match_req_ids(finished_ids, strip_suffix=False)
        finished_keys = self.request_helper.finalize_mediators(
            matched, finished_ids, self.nnsight_model
        )
        saves, removals = self.request_helper.collect_saves(matched, finished_keys)
        self.request_helper.cleanup_finished(finished_keys, removals)

import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.distributed.parallel_state import get_tp_group
from nnsight.intervention.tracing.globals import Globals

from ....modeling.common.request_helper import NNsightRequestHelper
from ..batching import VLLMBatcher

if TYPE_CHECKING:
    from ..vllm import VLLM
else:
    VLLM = Any

if TYPE_CHECKING:

    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput


class NNsightGPUModelRunner(GPUModelRunner):
    """Custom vLLM GPU model runner that interleaves NNsight interventions with model execution.

    Wraps the model with an NNsight :class:`Envoy`, deserializes
    mediators from incoming :class:`NNsightSamplingParams`, and manages
    batch group mappings so each invoke's intervention code sees the
    correct slice of the batch.
    """

    def __init__(self, *args, **kwargs):

        from .. import VLLM

        super().__init__(*args, **kwargs)

        self.nnsight_model: VLLM

        self.nnsight_request_helper = NNsightRequestHelper()

    def load_model(self, *args, **kwargs) -> None:

        from .. import VLLM

        super().load_model(*args, **kwargs)

        self.nnsight_model = VLLM(self.model)

        self.nnsight_model.tokenizer = cached_tokenizer_from_config(self.model_config)

        self.nnsight_model._interleaver.mediators = []

        self.nnsight_model._interleaver.batcher = VLLMBatcher()

        # Always call wrap() to register compat transforms (module
        # detection for HF-compatibility layer).  TP gather/split
        # hooks are gated on world_size > 1 inside wrap() itself.
        self.nnsight_model._interleaver.batcher.wrap(self.nnsight_model)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        super()._update_states(scheduler_output)

        self.nnsight_request_helper.process_new_reqs_serialized(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

        # Use input_batch.req_ids for the actual batch order after
        # condense()/reorder, not the scheduler dict order.
        # Store these for unflatten() which needs the same ordering.
        self.nnsight_request_helper._batch_req_ids = list(self.input_batch.req_ids)
        self.nnsight_request_helper._num_scheduled_tokens = dict(scheduler_output.num_scheduled_tokens)

        self.nnsight_request_helper.process_batch_groups(
            scheduler_output.num_scheduled_tokens,
            self.input_batch.req_ids,
            self.nnsight_model,
        )

        self.nnsight_model._interleaver.batcher.needs_batching = (
            len(self.nnsight_model._interleaver.mediators) > 1
        )

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):

        Globals.enter()

        return_value = None
        interleaver = self.nnsight_model._interleaver
        interleaver._defer_exceptions = True

        with interleaver:

            return_value = super().execute_model(scheduler_output, intermediate_tensors)

            self.nnsight_request_helper.unflatten(self.nnsight_model)

            if self.execute_model_state is not None:

                logits = self.nnsight_model.logits(
                    self.execute_model_state.logits, hook=True
                )

                state = self.execute_model_state

                self.execute_model_state = type(state)(
                    **{**state._asdict(), "logits": logits}
                )

        interleaver._defer_exceptions = False

        # Safety net: if __enter__ failed or forward was interrupted before
        # return_value could be assigned, build a minimal valid output.
        if return_value is None:
            from vllm.v1.outputs import ModelRunnerOutput
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            return_value = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            )

        Globals.exit()

        return return_value

    def _sample(self, *args, **kwargs):

        Globals.enter()

        sampler_output = None
        interleaver = self.nnsight_model._interleaver
        interleaver._defer_exceptions = True

        with interleaver:

            sampler_output = super()._sample(*args, **kwargs)

            sampler_output.sampled_token_ids = self.model.samples(
                sampler_output.sampled_token_ids, hook=True
            )

        interleaver._defer_exceptions = False

        Globals.exit()

        return sampler_output

    def collect_nnsight(
        self,
        req_ids: list[str],
        finished_req_ids: list[str] | None = None,
    ) -> Optional[bytes]:
        """Collect saved values from mediators, optionally finalizing finished requests.

        Called on every streamed output (async) or on finished requests (sync).
        Saves are collected for ALL ``req_ids``.  Mediators listed in
        ``finished_req_ids`` are additionally finalized (result handler,
        cancel) and cleaned up.

        Args:
            req_ids: Request IDs to collect current saves from.
            finished_req_ids: Subset of request IDs that are finished and
                should be finalized and cleaned up.  ``None`` means no
                requests are finished.
        """
        if get_pp_group().rank != 0:
            return None

        if finished_req_ids is None:
            finished_req_ids = []

        helper = self.nnsight_request_helper
        req_id_set = set(req_ids) | set(finished_req_ids)
        finished_req_id_set = set(finished_req_ids)

        matched = helper.match_req_ids(req_id_set)
        finished_keys = helper.finalize_mediators(
            matched, finished_req_id_set, self.nnsight_model
        )
        saves, removals = helper.collect_saves(matched, finished_keys)
        helper.cleanup_finished(finished_keys, removals)

        # Collect deferred exceptions from mediators, keyed by request ID.
        # Each invoke has its own mediator and its own exception.
        # The wrapped exception can't be pickled (dynamic class), so send type + message.
        exceptions = {}
        for base_id, mediator, internal_key in matched:
            if mediator._deferred_exception is not None:
                exc = mediator._deferred_exception
                exceptions[base_id] = {
                    "type": type(exc).__bases__[0].__name__ if hasattr(type(exc), "__bases__") else type(exc).__name__,
                    "message": str(exc),
                }
                mediator._deferred_exception = None

        if exceptions:
            saves["__nnsight_exceptions__"] = exceptions

        return pickle.dumps(saves)

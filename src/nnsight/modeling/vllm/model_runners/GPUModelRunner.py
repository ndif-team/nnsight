from typing import TYPE_CHECKING, Optional

from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from .runtime import NNsightRunnerMixin

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class NNsightGPUModelRunner(NNsightRunnerMixin, GPUModelRunner):
    """Interleave interventions with the original vLLM V1 model runner."""

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        super()._update_states(scheduler_output)

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

        # Use input_batch.req_ids for the actual batch order after
        # condense()/reorder, not the scheduler dict order.
        # Store these for unflatten() which needs the same ordering.
        self.nnsight_request_helper._batch_req_ids = list(self.input_batch.req_ids)
        self.nnsight_request_helper._num_scheduled_tokens = dict(
            scheduler_output.num_scheduled_tokens
        )

        self.nnsight_request_helper.process_batch_groups(
            scheduler_output.num_scheduled_tokens,
            self.input_batch.req_ids,
            self.nnsight_model,
        )

        self.nnsight_model.interleaver.batcher.needs_batching = (
            len(self.nnsight_model.interleaver.mediators) > 1
        )

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):

        return_value = None
        interleaver = self.nnsight_model.interleaver

        with interleaver:

            return_value = super().execute_model(scheduler_output, intermediate_tensors)

            self.nnsight_request_helper.unflatten(self.nnsight_model)

        # Safety net: if ``__enter__`` raised or the forward pass was
        # interrupted before ``return_value`` was assigned, ship back a
        # minimal valid ``ModelRunnerOutput`` so vLLM does not segfault.
        if return_value is None:
            from vllm.v1.outputs import ModelRunnerOutput

            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            return_value = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            )

        return return_value

    def sample_tokens(self, *args, **kwargs):

        interleaver = self.nnsight_model.interleaver

        with interleaver:

            # Provide logits from execute_model state before sampling.
            if self.execute_model_state is not None:

                logits = type(self.nnsight_model).logits.provide(
                    self.nnsight_model,
                    self.execute_model_state.logits,
                )

                state = self.execute_model_state

                self.execute_model_state = type(state)(
                    **{**state._asdict(), "logits": logits}
                )

        return super().sample_tokens(*args, **kwargs)

    def _sample(self, *args, **kwargs):

        sampler_output = None
        interleaver = self.nnsight_model.interleaver

        with interleaver:

            sampler_output = super()._sample(*args, **kwargs)

            sampler_output.sampled_token_ids = type(self.nnsight_model).samples.provide(
                self.nnsight_model,
                sampler_output.sampled_token_ids,
            )

        return sampler_output

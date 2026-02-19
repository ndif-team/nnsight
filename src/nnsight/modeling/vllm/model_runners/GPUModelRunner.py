import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from nnsight.intervention.tracing.globals import Globals

from ....intervention.serialization import load
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

    class NNsightRequestHelper:
        """
        Helper class for batching requests in the GPUModelRunner.

        Attributes:
            ids_to_batch_group (Dict[str, int]): Dictionary mapping request IDs to their assigned batch group indices.
            interleaver_to_ids (Dict[Interleaver, Set[str]]): Dictionary mapping interleavers to sets of request IDs.
            flat_batch_groups (Dict[Interleaver, List[Tuple[int, int]]]): Dictionary mapping interleavers to their flattened batch groups.

        Methods:
            process_new_reqs(new_reqs: List[NewRequestData]) -> None: Process new requests and compute the flat batch groups.
            process_finished_req(req_id: str, interleaver: Interleaver) -> None: Process a finished request,
                by updating batch groups and cleaning up mappings.
        """

        def __init__(self):

            self.req_id_to_batch_group_idx: Dict[str, int] = {}
            self.num_prompts_in_mediator = {}
            self.mediators: Dict[str, Any] = {}  # req_id -> Mediator
            self.last_mediator = None  # persists across process_new_reqs calls

        def process_new_reqs(
            self, new_reqs: List["NewRequestData"], model: VLLM
        ) -> None:
            """
            Process new requests and organize them into batch groups for execution.

            This method handles the batching logic for new requests, organizing them
            into appropriate batch groups based on their interleaver's batching strategy.

            Args:
                new_reqs (List[NewRequestData]): List of new request data objects to process.
                    Each request contains sampling parameters with an associated interleaver
                    that defines the batching behavior.

            Notes:
                - Resets the flat_batch_groups dictionary at the start
                - For interleavers that require batching, requests are assigned to batch groups
                - Batch groups are tuples of (start_position, size) indicating token ranges
                - Updates internal tracking dictionaries for request-to-batch-group mapping
                - Advances to next batch group when current group capacity is exceeded
            """

            for new_req in new_reqs:

                extra_args = getattr(new_req.sampling_params, 'extra_args', None)
                mediator_bytes = extra_args.get("nnsight_mediator") if extra_args else None

                # If its the first prompt / request within an invoke
                if mediator_bytes is not None:

                    self.last_mediator = load(
                        mediator_bytes,
                        model._remoteable_persistent_objects(),
                    )

                    model._interleaver.mediators.append(self.last_mediator)

                    self.last_mediator.start(model._interleaver)

                    self.num_prompts_in_mediator[self.last_mediator] = 1

                elif extra_args and extra_args.get("nnsight_batch_member"):

                    if self.last_mediator not in self.num_prompts_in_mediator:
                        self.num_prompts_in_mediator[self.last_mediator] = 0
                    self.num_prompts_in_mediator[self.last_mediator] += 1

                else:
                    # Non-NNsight request, skip
                    continue

                self.mediators[new_req.req_id] = self.last_mediator

        def unflatten(self, model: VLLM):

            batch_start = 0

            for mediator in model._interleaver.mediators:

                batch_group = mediator.batch_group

                batch_size = batch_group[1]

                if mediator in self.num_prompts_in_mediator:
                    batch_size = self.num_prompts_in_mediator[mediator]

                mediator.batch_group[0] = batch_start
                mediator.batch_group[1] = batch_size

                batch_start += batch_size

                model._interleaver.batcher.last_batch_group = mediator.batch_group

            self.num_prompts_in_mediator.clear()

        def process_batch_groups(
            self,
            num_tokens_scheduled: Dict[str, int],
            requests,
            model: VLLM,
        ) -> None:

            batch_start = 0

            seen_mediators = set()

            mediators = []

            for req_id, num_tokens in num_tokens_scheduled.items():

                mediator = self.mediators.get(req_id)

                if mediator is None:
                    batch_start += num_tokens
                    continue

                if mediator in seen_mediators:

                    mediator.batch_group[1] += num_tokens

                else:

                    seen_mediators.add(mediator)
                    mediators.append(mediator)
                    mediator.batch_group = [batch_start, num_tokens]

                batch_start += num_tokens

            if seen_mediators:
                model._interleaver.batcher.last_batch_group = mediator.batch_group
            else:
                model._interleaver.batcher.last_batch_group = None

            model._interleaver.mediators = mediators

    def __init__(self, *args, **kwargs):

        from .. import VLLM

        super().__init__(*args, **kwargs)

        self.nnsight_model: VLLM

        self.nnsight_request_helper = self.NNsightRequestHelper()

    def load_model(self, *args, **kwargs) -> None:

        from .. import VLLM

        super().load_model(*args, **kwargs)

        self.nnsight_model = VLLM(self.model)

        self.nnsight_model.tokenizer = cached_tokenizer_from_config(self.model_config)

        self.nnsight_model._interleaver.mediators = []

        self.nnsight_model._interleaver.batcher = VLLMBatcher()

        self.nnsight_model._interleaver.batcher.wrap(self.nnsight_model)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        super()._update_states(scheduler_output)

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

        self.nnsight_request_helper.process_batch_groups(
            scheduler_output.num_scheduled_tokens, self.requests, self.nnsight_model
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
        with self.nnsight_model._interleaver:

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

        Globals.exit()

        return return_value

    def _sample(self, *args, **kwargs):

        Globals.enter()

        with self.nnsight_model._interleaver:

            sampler_output = super()._sample(*args, **kwargs)

            sampler_output.sampled_token_ids = self.model.samples(
                sampler_output.sampled_token_ids, hook=True
            )

        Globals.exit()

        return sampler_output

    def finish_nnsight(
        self, finished_req_ids: list[str]
    ) -> Optional[bytes]:
        result = None

        finished_req_id_set = set(finished_req_ids)

        if get_pp_group().rank == 0:

            # Match finished engine-level req_ids to our stored mediators.
            # Use the mediators dict (keyed by internal req_id like "0-abc123")
            # since self.requests may already be cleaned up in multiprocessing mode.
            matched = []
            matched_keys = []

            for req_id, mediator in self.nnsight_request_helper.mediators.items():
                internal_id = req_id.split("-")[0]
                if internal_id in finished_req_id_set:
                    matched.append((internal_id, mediator))
                    matched_keys.append(req_id)

            outputs = []

            Globals.enter()

            for i, (internal_id, mediator) in enumerate(matched):

                outputs.append(internal_id)

                next_mediator = matched[i + 1][1] if i < len(matched) - 1 else None

                if next_mediator != mediator:

                    if mediator.alive:

                        self.nnsight_model._interleaver.mediators = [mediator]
                        mediator.batch_group = None

                        with self.nnsight_model._interleaver:
                            self.nnsight_model._interleaver.handle("result", outputs)

                            mediator.cancel()

                            self.nnsight_model._interleaver.handle()

                    outputs = []

            Globals.exit()

            saves = {}

            removals = set()

            for _, mediator in matched:
                frame = mediator.info.frame

                for key, value in frame.f_locals.items():

                    if id(value) in Globals.saves:
                        saves[key] = value
                        removals.add(id(value))

            for _id in removals:
                Globals.saves.remove(_id)

            # Pickle so it survives msgpack transport in multiprocessing mode
            result = pickle.dumps(saves)

            # Clean up mediator entries for finished requests
            for req_id in matched_keys:
                self.nnsight_request_helper.mediators.pop(req_id, None)

        return result

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from vllm.distributed.parallel_state import get_pp_group
from vllm.outputs import RequestOutput
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
from vllm.v1.outputs import ModelRunnerOutput
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

            last_mediator = None

            for new_req in new_reqs:

                if isinstance(new_req.sampling_params.mediator, bytes):

                    new_req.sampling_params.mediator = load(
                        new_req.sampling_params.mediator, model
                    )

                mediator = new_req.sampling_params.mediator

                batch_size = len(new_req.prompt_token_ids)

                # If its the first prompt / request within an invoke
                if mediator is not None:

                    last_mediator = mediator

                    model._interleaver.mediators.append(mediator)

                    mediator.start(model._interleaver)

                    batch_start = 0

                    if model._interleaver.batcher.last_batch_group:
                        batch_start = sum(model._interleaver.batcher.last_batch_group)

                    batch_group = [
                        batch_start,
                        batch_size,
                    ]
                    mediator.batch_group = batch_group

                    model._interleaver.batcher.last_batch_group = batch_group

                    self.num_prompts_in_mediator[mediator] = 1

                else:

                    new_req.sampling_params.mediator = last_mediator

                    last_flattened_batch_group = last_mediator.batch_group

                    last_flattened_batch_group[1] += batch_size

                    self.num_prompts_in_mediator[last_mediator] += 1

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

            self.num_prompts_in_mediator.clear()

        def process_finished_reqs(
            self, finished_request_ids: Set[str], requests, model: VLLM
        ) -> None:

            batch_start = 0

            seen_mediators = set()

            for req_id, req in requests.items():

                mediator = req.sampling_params.mediator

                if req_id in finished_request_ids:

                    continue

                if mediator in seen_mediators:

                    mediator.batch_group[1] += 1

                else:

                    seen_mediators.add(mediator)

                    mediator.batch_group[0] = batch_start
                    mediator.batch_group[1] = 1

                batch_start += 1

            if seen_mediators:
                model._interleaver.batcher.last_batch_group = mediator.batch_group
            else:
                model._interleaver.batcher.last_batch_group = None

    def __init__(self, *args, **kwargs):

        from .. import VLLM

        super().__init__(*args, **kwargs)

        self.nnsight_model: VLLM

        self.nnsight_request_helper = self.NNsightRequestHelper()

    def load_model(self, *args, **kwargs) -> None:

        from .. import VLLM

        super().load_model(*args, **kwargs)

        self.nnsight_model = VLLM(self.model)

        self.nnsight_model.tokenizer = init_tokenizer_from_configs(self.model_config)

        self.nnsight_model._interleaver.mediators = []

        self.nnsight_model._interleaver.batcher = VLLMBatcher()

        self.nnsight_model._interleaver.batcher.wrap(self.nnsight_model)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

        self.nnsight_model._interleaver.batcher.needs_batching = (
            len(self.nnsight_model._interleaver.mediators) > 1
        )

        return super()._update_states(scheduler_output)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):

        Globals.enter()
        with self.nnsight_model._interleaver:

            super().execute_model(scheduler_output, intermediate_tensors)

            self.nnsight_request_helper.unflatten(self.nnsight_model)

            logits = self.model.logits(self.execute_model_state.logits, hook=True)

            state = self.execute_model_state

            self.execute_model_state = type(state)(
                **{**state._asdict(), "logits": logits}
            )

        Globals.exit()

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
        self, finished_requests: list[RequestOutput]
    ) -> ModelRunnerOutput:

        result = None

        # TODO this might not be the output rank?
        if get_pp_group().rank == 0:

            Globals.enter()
            with self.nnsight_model._interleaver:
                finished_requests[0] = self.nnsight_model._interleaver.handle(
                    "result", finished_requests[0]
                )

            Globals.exit()

            result = {}

            removals = set()

            # TODO saves need to be removed on all processes
            for req in finished_requests:
                req = self.requests[req.request_id]
                if req.sampling_params.mediator is None:
                    continue
                frame = req.sampling_params.mediator.info.frame

                for key, value in frame.items():

                    if id(value) in Globals.saves:
                        result[key] = value
                        removals.add(id(value))

            for _id in removals:
                Globals.saves.remove(_id)

        finished_req_ids = set([req.request_id for req in finished_requests])

        with self.nnsight_model._interleaver:

            for req_id in finished_req_ids:
                req = self.requests[req_id]
                req.sampling_params.mediator.cancel()

            self.nnsight_model._interleaver.handle()

        self.nnsight_request_helper.process_finished_reqs(
            finished_req_ids, self.requests, self.nnsight_model
        )

        return result

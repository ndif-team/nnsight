from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Set

from vllm.distributed.parallel_state import get_pp_group
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs

from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from ....intervention.serialization import load
from vllm.outputs import RequestOutput
from nnsight.intervention.tracing.globals import Globals
from ..batching import VLLMBatcher

if TYPE_CHECKING:

    from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData

    from ....intervention.interleaver import Interleaver


class NNsightGPUModelRunner(GPUModelRunner):

    class NNsightRequestHelper:
        '''
        Helper class for batching requests in the GPUModelRunner.

        Attributes:
            ids_to_batch_group (Dict[str, int]): Dictionary mapping request IDs to their assigned batch group indices.
            interleaver_to_ids (Dict[Interleaver, Set[str]]): Dictionary mapping interleavers to sets of request IDs.
            flat_batch_groups (Dict[Interleaver, List[Tuple[int, int]]]): Dictionary mapping interleavers to their flattened batch groups.

        Methods:
            process_new_reqs(new_reqs: List[NewRequestData]) -> None: Process new requests and compute the flat batch groups.
            process_finished_req(req_id: str, interleaver: Interleaver) -> None: Process a finished request,
                by updating batch groups and cleaning up mappings.
        '''

        def __init__(self):

            self.ids_to_batch_group: Dict[str, int] = {}
            self.interleaver_to_ids: Dict["Interleaver",
                                          Set[str]] = defaultdict(set)
            self.flat_batch_groups: Dict["Interleaver",
                                         List[Tuple[int, int]]] = defaultdict(list)

        def process_new_reqs(self, new_reqs: List["NewRequestData"], model) -> None:
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

            # reset
            self.flat_batch_groups = defaultdict(list)

            # current batch group index per interleaver
            curr_batch_groups: Dict["Interleaver", int] = defaultdict(int)

            interleavers: Set["Interleaver"] = set()

            for new_req in new_reqs:

                if isinstance(new_req.sampling_params.interleaver, bytes):

                    new_req.sampling_params.interleaver = load(
                        new_req.sampling_params.interleaver, model)

                interleaver = new_req.sampling_params.interleaver

                model._interleaver.invokers.extend(interleaver.invokers)
                model._interleaver.asynchronous = interleaver.asynchronous
                for invoker in interleaver.invokers:
                    invoker.start(model._interleaver)

                # clean up any finished requests from the previous round
                if interleaver not in interleavers:
                    interleavers.add(interleaver)

                    if interleaver in self.interleaver_to_ids:
                        for req_id in self.interleaver_to_ids[interleaver]:
                            self.ids_to_batch_group.pop(req_id)

                    self.interleaver_to_ids[interleaver] = set()

                if interleaver.batcher.needs_batching:
                    # linking request to batch group
                    batch_groups: List[Tuple[int, int]
                                       ] = interleaver.batcher.batch_groups
                    batch_group_idx: int = curr_batch_groups[interleaver]

                    self.interleaver_to_ids[interleaver].add(new_req.req_id)

                    # update current batch group index
                    if len(self.interleaver_to_ids[interleaver]) > sum(batch_groups[batch_group_idx]):
                        curr_batch_groups[interleaver] += 1

                    self.ids_to_batch_group[new_req.req_id] = curr_batch_groups[interleaver]

                    # first seen request
                    if self.flat_batch_groups[interleaver] == []:
                        # req_to_interleaver_dict[interleaver] = (0, 0)
                        # the first batch group starts at 0 and has a size of the number of tokens in the prompt
                        batch_group = (batch_groups[0][0], len(
                            new_req.prompt_token_ids))
                        self.flat_batch_groups[interleaver].append(batch_group)

                    else:
                        req_count: int = len(
                            self.interleaver_to_ids[interleaver])
                        curr_batch_group: Tuple[int,
                                                int] = batch_groups[batch_group_idx]
                        last_flat_batch_group: Tuple[int,
                                                     int] = self.flat_batch_groups[interleaver][-1]

                        # check if the current request is within the current batch group
                        if req_count < sum(curr_batch_group):
                            # update the current batch group with the number of tokens in the new request
                            batch_group = (
                                last_flat_batch_group[0], last_flat_batch_group[1] + len(new_req.prompt_token_ids))
                            self.flat_batch_groups[interleaver][-1] = batch_group

                        else:
                            # add a new batch group
                            batch_group = (sum(last_flat_batch_group), len(
                                new_req.prompt_token_ids))
                            self.flat_batch_groups[interleaver].append(
                                batch_group)

        def process_finished_req(self, req_id: str, interleaver: "Interleaver") -> None:
            """
            Process a finished request by updating batch groups and cleaning up mappings.

            This method handles the removal of a completed request from the batch tracking
            system. When a request finishes, it:
            1. Removes the request from the interleaver's tracking set
            2. Decrements the batch group count 
            3. If the batch group becomes empty, removes it and updates indices
            4. Updates batch group mappings for remaining requests
            5. Cleans up the request ID mapping

            Args:
                req_id (str): The unique identifier of the finished request
                interleaver (Interleaver): The interleaver instance managing this request
            """

            if req_id in self.interleaver_to_ids[interleaver]:

                # remove finished request
                self.interleaver_to_ids[interleaver].remove(req_id)

                batch_idx = self.ids_to_batch_group[req_id]
                batch_groups = interleaver.batcher.batch_groups

                batch_group = batch_groups[batch_idx]
                new_batch_group = (batch_group[0], batch_group[1] - 1)

                if new_batch_group[1] == 0:
                    batch_groups.pop(batch_idx)

                    for idx in range(batch_idx, len(batch_groups)):
                        # update the batch start
                        if idx == batch_idx:
                            batch_groups[idx] = (
                                batch_group[0], batch_groups[idx][1])
                        else:
                            batch_groups[idx] = (
                                sum(batch_groups[idx-1]), batch_groups[idx][1])

                    # update the batch group index for remaining requests
                    for other_req_id in self.interleaver_to_ids[interleaver]:
                        if self.ids_to_batch_group[other_req_id] > batch_idx:
                            self.ids_to_batch_group[other_req_id] -= 1

                    for mediator in interleaver.invokers:
                        if mediator.batch_group and mediator.batch_group > batch_idx:
                            mediator.batch_group -= 1

                        if mediator.child:
                            mediator.child.batch_group = mediator.batch_group

                    interleaver.batcher.batch_groups = batch_groups

                else:
                    batch_groups[batch_idx] = new_batch_group

                # remove finished request
                self.ids_to_batch_group.pop(req_id)

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

        self.nnsight_model._interleaver.invokers = []

        self.nnsight_model._interleaver.batcher = VLLMBatcher()

        self.nnsight_model._interleaver.batcher.wrap(self.nnsight_model)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model)

        for req_id in scheduler_output.finished_req_ids:

            self.nnsight_request_helper.process_finished_req(
                req_id,
                self.requests[req_id].sampling_params.interleaver
            )

        for new_req_data in scheduler_output.scheduled_new_reqs:

            sampling_params = new_req_data.sampling_params

            batcher = sampling_params.interleaver.batcher
            batcher.cache_batch_groups(
                self.nnsight_request_helper.flat_batch_groups[sampling_params.interleaver])

        return super()._update_states(scheduler_output)

    def execute_model(self,
                      scheduler_output: "SchedulerOutput",
                      intermediate_tensors: Optional[IntermediateTensors] = None
                      ):

        Globals.enter()

        with self.nnsight_model._interleaver:

            super().execute_model(scheduler_output, intermediate_tensors)

            self.nnsight_model._interleaver.batcher.restore_batch_groups()

            logits = self.model.logits(
                self.execute_model_state.logits, hook=True)

            state = self.execute_model_state

            self.execute_model_state = type(state)(
                **{**state._asdict(), "logits": logits})

        Globals.exit()

    def finish_nnsight(self, finished_requests: list[RequestOutput]) -> ModelRunnerOutput:

        result = None

        # TODO this might not be the output rank?
        if get_pp_group().rank == 0:

            Globals.enter()

            with self.nnsight_model._interleaver:
                finished_requests[0] = self.nnsight_model._interleaver.handle(
                    'result', finished_requests[0])

            Globals.exit()

            result = list(self.requests.values())[0].sampling_params.interleaver.invokers[0].info.frame

            result = {key: value for key,
                      value in result.items() if id(value) in Globals.saves}

            Globals.saves.clear()

        self.nnsight_model._interleaver.invokers = [
            invoker for invoker in self.nnsight_model._interleaver.invokers if invoker.alive]

        self.nnsight_model._interleaver.mediators.clear()
        self.nnsight_model._interleaver.iteration_tracker.clear()

        return result

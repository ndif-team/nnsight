from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

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

    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

    from ....intervention.interleaver import Interleaver


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
            self.num_prompts_in_batch_group = {}

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
            
            last_mediator = None

            for new_req in new_reqs:

                if isinstance(new_req.sampling_params.mediator, bytes):

                    new_req.sampling_params.mediator = load(
                        new_req.sampling_params.mediator, model
                    )

                mediator = new_req.sampling_params.mediator
                            
                batch_size = len(new_req.prompt_token_ids)
                batch_group_idx = len(model._interleaver.batcher.batch_groups)
                
                mediator.batch_group = batch_group_idx
                
                model._interleaver.batcher.needs_batching = True
                
                self.req_id_to_batch_group_idx[new_req.req_id] = batch_group_idx
                
                # If its the first prompt / request within an invoke
                if mediator is not None:
                    
                    last_mediator = mediator

                    # Start the intervention thread
                    mediator.start(model._interleaver)
                    
                    # If the internvetniion thread didnt immediately complete, add it to the invokers
                    if mediator.alive:
                        
                        model._interleaver.invokers.append(mediator)
                    
                    batch_start = 0
                
                    if model._interleaver.batcher.batch_groups:
                        batch_start = sum(model._interleaver.batcher.batch_groups[-1])

                    batch_group = (
                        batch_start,
                        batch_size,
                    )
                    model._interleaver.batcher.batch_groups.append(batch_group)
                    
                    self.num_prompts_in_batch_group[batch_group_idx] = 1
                                            
                else:
                    
                    new_req.sampling_params.mediator = last_mediator
                    
                    last_flattened_batch_group = model._interleaver.batcher.batch_groups[-1]
                    
                    batch_group = (
                        last_flattened_batch_group[0],
                        last_flattened_batch_group[1]
                        + batch_size,
                    )
                    model._interleaver.batcher.batch_groups[-1] = batch_group
                    
                    self.num_prompts_in_batch_group[batch_group_idx] += 1
                                        
        def unflatten(self, model):
            
            
            batch_start = 0
            
            for batch_group_idx in range(len(model._interleaver.batcher.batch_groups)):
                
                batch_group = model._interleaver.batcher.batch_groups[batch_group_idx]
                
                batch_size = batch_group[1]
                
                if batch_group_idx in self.num_prompts_in_batch_group:
                    batch_size = self.num_prompts_in_batch_group[batch_group_idx]
                    
                model._interleaver.batcher.batch_groups[batch_group_idx] = (batch_start, batch_size)
                
                batch_start += batch_size
                
            self.num_prompts_in_batch_group.clear()
            
        def process_finished_reqs(self, finished_request_ids: Set[str], requests, model) -> None:
            
            batch_start = 0
            
            batch_groups = []
            
            seen_mediators = set()
            
            for req_id, req in requests.items():
                
                mediator = req.sampling_params.mediator
                
                if req_id in finished_request_ids:
                    
                    continue
                
                if mediator in seen_mediators:
                    
                    batch_group = batch_groups[-1]
                    
                    batch_groups[-1] = (batch_group[0], batch_group[1] + 1)
                    
                else:
                    
                    seen_mediators.add(mediator)
                    
                    mediator.batch_group = len(batch_groups)
                    
                    batch_groups.append((batch_start, 1))
                    
                batch_start += 1
                    
            model._interleaver.batcher.batch_groups = batch_groups
                    
                    
                
                
                

                
                
                
                
                
                
                
                
            
           

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

        self.nnsight_model._interleaver.batcher.cached_batch_groups = []

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
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

        self.nnsight_model._interleaver.invokers = [
            invoker
            for invoker in self.nnsight_model._interleaver.invokers
            if invoker.alive
        ]

    def _sample(self, *args, **kwargs):

        Globals.enter()
        with self.nnsight_model._interleaver:

            sampler_output = super()._sample(*args, **kwargs)

            sampler_output.sampled_token_ids = self.model.samples(
                sampler_output.sampled_token_ids, hook=True
            )

        Globals.exit()

        self.nnsight_model._interleaver.invokers = [
            invoker
            for invoker in self.nnsight_model._interleaver.invokers
            if invoker.alive
        ]

        return sampler_output

    def finish_nnsight(
        self, finished_requests: list[RequestOutput]
    ) -> ModelRunnerOutput:

        result = None
        
        print("FIINISH NNSIGHT")

        # TODO this might not be the output rank?
        if get_pp_group().rank == 0:

            Globals.enter()
            with self.nnsight_model._interleaver:
                finished_requests[0] = self.nnsight_model._interleaver.handle(
                    "result", finished_requests[0]
                )

            Globals.exit()

            result = {}

            for req in finished_requests:
                req = self.requests[req.request_id]
                if req.sampling_params.mediator is None:
                    continue
                frame = req.sampling_params.mediator.info.frame

                for key, value in frame.items():
                    if id(value) in Globals.saves:
                        result[key] = value
                        Globals.saves.remove(id(value))

        self.nnsight_model._interleaver.invokers = [
            invoker
            for invoker in self.nnsight_model._interleaver.invokers
            if invoker.alive
        ]

        return result

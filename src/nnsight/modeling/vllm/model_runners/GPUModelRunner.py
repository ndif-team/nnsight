from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Set

import torch
import torch.distributed

import vllm.envs as envs
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import LazyLoader, round_up
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:

    from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData

    from ....intervention.interleaver import Interleaver
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

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
            self.interleaver_to_ids: Dict["Interleaver", Set[str]] = defaultdict(set)
            self.flat_batch_groups: Dict["Interleaver", List[Tuple[int, int]]] = defaultdict(list)

        def process_new_reqs(self, new_reqs: List["NewRequestData"]) -> None:
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
                interleaver = new_req.sampling_params.interleaver

                # clean up any finished requests from the previous round
                if interleaver not in interleavers:
                    interleavers.add(interleaver)

                    if interleaver in self.interleaver_to_ids:
                        for req_id in self.interleaver_to_ids[interleaver]:
                            self.ids_to_batch_group.pop(req_id)

                    self.interleaver_to_ids[interleaver] = set()

                if interleaver.batcher.needs_batching:
                    # linking request to batch group
                    batch_groups: List[Tuple[int, int]] = interleaver.batcher.batch_groups
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
                        batch_group = (batch_groups[0][0], len(new_req.prompt_token_ids))
                        self.flat_batch_groups[interleaver].append(batch_group)

                    else:
                        req_count: int = len(self.interleaver_to_ids[interleaver])
                        curr_batch_group: Tuple[int, int] = batch_groups[batch_group_idx]
                        last_flat_batch_group: Tuple[int, int] = self.flat_batch_groups[interleaver][-1]

                        # check if the current request is within the current batch group
                        if req_count < sum(curr_batch_group):
                            # update the current batch group with the number of tokens in the new request
                            batch_group = (last_flat_batch_group[0], last_flat_batch_group[1] + len(new_req.prompt_token_ids))
                            self.flat_batch_groups[interleaver][-1] = batch_group

                        else:
                            # add a new batch group
                            batch_group = (sum(last_flat_batch_group), len(new_req.prompt_token_ids))
                            self.flat_batch_groups[interleaver].append(batch_group)

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
                            batch_groups[idx] = (batch_group[0], batch_groups[idx][1])
                        else:
                            batch_groups[idx] = (sum(batch_groups[idx-1]), batch_groups[idx][1])

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

    def load_model(self) -> None:
        
        from .. import VLLM

        super().load_model()

        self.nnsight_model = VLLM(self.model)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """

        ########## NNsight ##########

        self.nnsight_request_helper.process_new_reqs(scheduler_output.scheduled_new_reqs)

        #############################

        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            
            ########## NNsight ##########

            self.nnsight_request_helper.process_finished_req(
                req_id, 
                self.requests[req_id].sampling_params.interleaver
            )

            #############################

            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        req_ids_to_add: list[str] = []

        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            ########## NNsight ##########

            # flatten batch groups to match vLLM's batch processing method
            batcher = sampling_params.interleaver.batcher
            batcher.cache_batch_groups(self.nnsight_request_helper.flat_batch_groups[sampling_params.interleaver])

            #############################

            pooling_params = new_req_data.pooling_params
            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.extend(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()


    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:

        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)

        # Prepare the decoder inputs.
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata,
         num_scheduled_tokens_np) = (self._prepare_inputs(scheduler_output))
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Some attention backends only support CUDA Graphs in pure decode.
        # If attention doesn't support CUDA Graphs for this batch, but we
        # compiled with full CUDA graphs, we have to skip them entirely.
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

        ### NNsight ####
        from contextlib import ExitStack

        interleavers = set([request.sampling_params.interleaver for request in  self.requests.values()])
        
        with ExitStack() as stack:
            for interleaver in interleavers:
                stack.enter_context(interleaver)

            # Run the model.
            # Use persistent buffers for CUDA graphs.
            with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    skip_cuda_graphs=skip_cuda_graphs,
            ):
                self.maybe_setup_kv_connector(scheduler_output)

                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

                self.maybe_wait_for_kv_save()
                finished_sending, finished_recving = (
                    self.get_finished_kv_transfers(scheduler_output))
            

            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = \
                self.parallel_config.distributed_executor_backend \
                == "external_launcher" and len(get_pp_group().ranks) > 0
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(hidden_states.tensors,
                                                all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    return self._pool(hidden_states, num_scheduled_tokens,
                                    num_scheduled_tokens_np, finished_sending,
                                    finished_recving)

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
            if broadcast_pp_output:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                } if logits is not None else {}
                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)


            #### NNsight ####

            # resetting the batch_groups on the interleaver's batcher
            for interleaver in interleavers:
                interleaver.batcher.restore_batch_groups()

            logits = self.model.logits(logits, hook=True)

            ###############


            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                assert logits is not None
                bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids


            ##### NNsight #####

            samples = self.model.samples(sampler_output.sampled_token_ids, hook=True)

            ###################

        ###################


        ####### NNsight #######

        for interleaver in interleavers:
                interleaver.invokers = [invoker for invoker in interleaver.invokers if invoker.alive]

        #######################


        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        if not self.speculative_config:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        else:
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                attn_metadata,
            )

        # Clear KVConnector state after all KVs are generated.
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            num_nans_in_logits=num_nans_in_logits,
        )


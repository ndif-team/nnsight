from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Set, cast

import torch
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.structured_output.utils import apply_grammar_bitmask

from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context, BatchDescriptor
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import LazyLoader
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, AsyncGPUModelRunnerOutput
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling )
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

    def load_model(self, *args, **kwargs) -> None:
        
        from .. import VLLM

        super().load_model(*args, **kwargs)

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


        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

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

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            ########## NNsight ##########

            # flatten batch groups to match vLLM's batch processing method
            batcher = sampling_params.interleaver.batcher
            batcher.cache_batch_groups(self.nnsight_request_helper.flat_batch_groups[sampling_params.interleaver])

            #############################

            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            reqs_to_add.append(req_state)

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
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

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
        for request in reqs_to_add:
            self.input_batch.add_request(request)

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

        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs")

                # Prepare the decoder inputs.
                (attn_metadata, logits_indices, spec_decode_metadata,
                 num_scheduled_tokens_np, spec_decode_common_attn_metadata,
                 max_query_len, ubatch_slices, num_tokens_after_padding
                 ) = self._prepare_inputs(scheduler_output)

            (
                num_scheduled_tokens,
                num_input_tokens,
                num_tokens_across_dp,
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
            ) = self._preprocess(scheduler_output, intermediate_tensors,
                                 ubatch_slices, num_tokens_after_padding)

            uniform_decode = (max_query_len
                              == self.uniform_decode_query_len) and (
                                  num_scheduled_tokens
                                  == self.input_batch.num_reqs * max_query_len)
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=uniform_decode)
            cudagraph_runtime_mode, batch_descriptor = \
                self.cudagraph_dispatcher.dispatch(batch_descriptor)

        # This is currently to get around the assert in the DPMetadata
        # where it wants `num_tokens_across_dp` to align with `num_tokens`
        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens

        ### NNsight ####
        from contextlib import ExitStack

        interleavers = set([request.sampling_params.interleaver for request in  self.requests.values()])
        
        with ExitStack() as stack:
            for interleaver in interleavers:
                stack.enter_context(interleaver)

            # Run the model.
            # Use persistent buffers for CUDA graphs.
            with (set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
            ), record_function_or_nullcontext("Forward"),
                self.maybe_get_kv_connector_output(scheduler_output) as
                kv_connector_output):
                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

    

            with record_function_or_nullcontext("Postprocess"):
                if self.use_aux_hidden_state_outputs:
                    # True when EAGLE 3 is used.
                    hidden_states, aux_hidden_states = model_output
                else:
                    # Common case.
                    hidden_states = model_output
                    aux_hidden_states = None

                if not self.broadcast_pp_output:
                    # Common case.
                    if not get_pp_group().is_last_rank:
                        # Return the intermediate tensors.
                        assert isinstance(hidden_states, IntermediateTensors)
                        hidden_states.kv_connector_output = kv_connector_output
                        return hidden_states

                    if self.is_pooling_model:
                        # Return the pooling output.
                        output = self._pool(hidden_states, num_scheduled_tokens,
                                            num_scheduled_tokens_np)
                        output.kv_connector_output = kv_connector_output
                        return output

                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)
                else:
                    # Rare case.
                    assert not self.is_pooling_model

                    if not get_pp_group().is_last_rank:
                        all_gather_tensors = {
                            "residual":
                            not is_residual_scattered_for_sp(
                                self.vllm_config, num_input_tokens)
                        }
                        get_pp_group().send_tensor_dict(
                            hidden_states.tensors,
                            all_gather_group=get_tp_group(),
                            all_gather_tensors=all_gather_tensors)
                        logits = None
                    else:
                        sample_hidden_states = hidden_states[logits_indices]
                        logits = self.model.compute_logits(sample_hidden_states)

                    model_output_broadcast_data = {}
                    if logits is not None:
                        model_output_broadcast_data["logits"] = logits.contiguous()

                    model_output_broadcast_data = get_pp_group(
                    ).broadcast_tensor_dict(model_output_broadcast_data,
                                            src=len(get_pp_group().ranks) - 1)
                    assert model_output_broadcast_data is not None
                    logits = model_output_broadcast_data["logits"]
                    
                    # Apply structured output bitmasks if present
                if scheduler_output.grammar_bitmask is not None:
                    apply_grammar_bitmask(scheduler_output, self.input_batch,
                                        logits, self.device)


            #### NNsight ####

            # resetting the batch_groups on the interleaver's batcher
            for interleaver in interleavers:
                interleaver.batcher.restore_batch_groups()

            logits = self.model.logits(logits, hook=True)

        ###############


        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("Draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        use_padded_batch_for_eagle = self.speculative_config and \
            self.speculative_config.use_eagle() and \
            not self.speculative_config.disable_padded_drafter_batch
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (self.speculative_config
                and self.speculative_config.draft_model_config is not None
                and self.speculative_config.draft_model_config.max_model_len
                is not None):
            effective_drafter_max_model_len = (
                self.speculative_config.draft_model_config.max_model_len)
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.seq_lens.max() +
            self.speculative_config.num_speculative_tokens
            <= effective_drafter_max_model_len)
        if use_padded_batch_for_eagle and input_fits_in_drafter:
            # EAGLE speculative decoding can use the GPU sampled tokens
            # as inputs, and does not need to wait for bookkeeping to finish.
            propose_draft_token_ids(sampler_output.sampled_token_ids)

        with record_function_or_nullcontext("Bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(scheduler_output, sampler_output,
                                       logits, hidden_states,
                                       num_scheduled_tokens)

        if (self.speculative_config and not use_padded_batch_for_eagle
                and input_fits_in_drafter):
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("EPLB"):
            self.eplb_step()
            
            
        ##### NNsight #####
            
        for interleaver in interleavers:
            interleaver.invokers = [invoker for invoker in interleaver.invokers if invoker.alive]
        #######################

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

        if not self.use_async_scheduling:
            return output

        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )



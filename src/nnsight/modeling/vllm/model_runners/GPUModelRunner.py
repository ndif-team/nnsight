import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from nnsight import NNsight
import torch
import torch.distributed

from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MultiModalInputs
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)
from vllm.worker.model_runner_base import (
    _add_attn_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict, dump_input_when_exception)

from ....intervention.interleaver import Interleaver
from .. import VLLM
from ..sampling import NNsightSamplingMetadata

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

    from ..sampling import NNsightSamplingMetadata


class NNsightModelInputForGPUWithSamplingMetadata(ModelInputForGPUWithSamplingMetadata):

    sampling_metadata: Optional["NNsightSamplingMetadata"] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "prompt_adapter_mapping": self.prompt_adapter_mapping,
            "prompt_adapter_requests": self.prompt_adapter_requests,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        if self.sampling_metadata is not None:
            tensor_dict["selected_token_indices"] = (
                self.sampling_metadata.selected_token_indices)
            tensor_dict["intervention_graph"] = self.sampling_metadata.intervention_graph
            tensor_dict["call_counter"] = self.sampling_metadata.call_counter
            tensor_dict["batch_group"] = self.sampling_metadata.batch_groups
        return tensor_dict
    
    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForGPUWithSamplingMetadata":
        selected_token_indices = tensor_dict.pop("selected_token_indices", None)
        intervention_graph = tensor_dict.pop("intervention_graph", None)
        intervention_graph.attachments = dict()
        call_counter = tensor_dict.pop("call_counter", None)
        batch_groups = tensor_dict.pop("batch_group", None)
        if selected_token_indices is not None:
            tensor_dict["sampling_metadata"] = NNsightSamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
                intervention_graph=intervention_graph,
                call_counter=call_counter,
                batch_groups=batch_groups
            )
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class NNsightGPUModelRunner(ModelRunner):

    _model_input_cls: Type[NNsightModelInputForGPUWithSamplingMetadata] = (
        NNsightModelInputForGPUWithSamplingMetadata
    )

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.model: VLLM

    def load_model(self) -> None:
        super().load_model()

        self.model = VLLM(self.model)

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> NNsightModelInputForGPUWithSamplingMetadata:
        model_input = \
            NNsightModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> NNsightModelInputForGPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids
        )
        if get_pp_group().is_last_rank:
            # Sampling metadata is only required for the final pp group
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = NNsightSamplingMetadata.prepare(
                seq_group_metadata_list,
                model_input.seq_lens,
                model_input.query_lens,
                self.device,
                self.pin_memory,
                generators,
                self.sampling_metadata_cache,
            )
        else:
            sampling_metadata = None
        is_prompt = (
            seq_group_metadata_list[0].is_prompt if seq_group_metadata_list else None
        )
        return dataclasses.replace(
            model_input,
            sampling_metadata=sampling_metadata,
            is_prompt=is_prompt,
            virtual_engine=virtual_engine,
        )

    @torch.inference_mode()
    @dump_input_when_exception(exclude_args=[0], exclude_kwargs=["self"])
    def execute_model(
        self,
        model_input: NNsightModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:

        if model_input.sampling_metadata.intervention_graph is None:

            return super().execute_model(
                model_input,
                kv_caches,
                intermediate_tensors=intermediate_tensors,
                num_steps=num_steps,
            )

        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests, model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests, model_input.prompt_adapter_mapping
            )

        self.attn_state.begin_forward(model_input)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][graph_batch_size]
        else:
            model_executable = self.model

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = (
            {
                "finished_requests_ids": model_input.finished_requests_ids,
                "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
            }
            if self.has_seqlen_agnostic
            else {}
        )
        if (
            self.observability_config is not None
            and self.observability_config.collect_model_forward_time
        ):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        ## NNSIGHT #########################################

        intervention_graph = model_input.sampling_metadata.intervention_graph

        batch_groups = model_input.sampling_metadata.batch_groups

        interleaver = Interleaver(
            intervention_graph, batch_groups=batch_groups, batch_size=len(model_input.input_tokens)
        )

        def inner():

            with set_forward_context(model_input.attn_metadata):
                hidden_or_intermediate_states = self.model._model(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    kv_caches=kv_caches,
                    attn_metadata=model_input.attn_metadata,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalInputs.as_kwargs(
                        multi_modal_kwargs, device=self.device
                    ),
                    **seqlen_agnostic_kwargs,
                )

            if (
                self.observability_config is not None
                and self.observability_config.collect_model_forward_time
            ):
                model_forward_end.record()

            # Compute the logits in the last pipeline stage.
            if not get_pp_group().is_last_rank:
                if (
                    self.is_driver_worker
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states, IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time
                ):
                    model_forward_end.synchronize()
                    model_forward_time = model_forward_start.elapsed_time(
                        model_forward_end
                    )
                    orig_model_forward_time = 0.0
                    if intermediate_tensors is not None:
                        orig_model_forward_time = intermediate_tensors.tensors.get(
                            "model_forward_time", torch.tensor(0.0)
                        ).item()
                    hidden_or_intermediate_states.tensors["model_forward_time"] = (
                        torch.tensor(model_forward_time + orig_model_forward_time)
                    )
                return hidden_or_intermediate_states

            logits = self.model.compute_logits(
                hidden_or_intermediate_states, model_input.sampling_metadata
            )

            logits = self.model.logits(logits)

            if not self.is_driver_worker:
                return []

            if model_input.async_callback is not None:
                model_input.async_callback()

            # Sample the next token.
            output: SamplerOutput = self.model.sample(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )
            output.sampled_token_ids = self.model.tokens(output.sampled_token_ids)
            if (
                self.observability_config is not None
                and self.observability_config.collect_model_forward_time
                and output is not None
            ):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)
                    ).item()
                # If there are multiple workers, we are still tracking the latency
                # from the start time of the driver worker to the end time of the
                # driver worker. The model forward time will then end up covering
                # the communication time as well.
                output.model_forward_time = orig_model_forward_time + model_forward_time

            if self.return_hidden_states:
                # we only need to pass hidden states of most recent token
                assert model_input.sampling_metadata is not None
                indices = model_input.sampling_metadata.selected_token_indices
                if model_input.is_prompt:
                    hidden_states = hidden_or_intermediate_states.index_select(
                        0, indices
                    )
                    output.prefill_hidden_states = hidden_or_intermediate_states
                elif decode_meta.use_cuda_graph:
                    hidden_states = hidden_or_intermediate_states[: len(indices)]
                else:
                    hidden_states = hidden_or_intermediate_states

                output.hidden_states = hidden_states

            return output

        output = NNsight.interleave(
            self.model,
            fn=inner,
            interleaver=interleaver,
        )
        return [output]
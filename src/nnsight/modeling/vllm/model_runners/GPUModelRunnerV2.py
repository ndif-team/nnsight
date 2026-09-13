"""NNsight adapter for vLLM 0.29's GPU Model Runner V2."""

from importlib.metadata import version

import numpy as np
import torch
from packaging.version import Version
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler

from .runtime import NNsightRunnerMixin


class NNsightGPUModelRunnerV2(NNsightRunnerMixin, GPUModelRunner):
    """Expose scheduled forwards, raw logits, and samples to interventions.

    Iterations count scheduled executions, including nonfinal prefill chunks.
    vLLM's ``num_sampled`` still determines whether a sample is emitted.
    """

    @staticmethod
    def validate_config(config):
        installed = Version(version("vllm"))
        if not Version("0.29.0") <= installed < Version("0.30.0"):
            raise ValueError(
                "NNsight Model Runner V2 requires vLLM >=0.29.0,<0.30.0; "
                f"found {installed}. Use VLLM_USE_V2_MODEL_RUNNER=0 for the "
                "legacy adapter on its supported vLLM versions."
            )

        model = config.model_config
        parallel = config.parallel_config
        unsupported = []
        if not model.enforce_eager:
            unsupported.append("CUDA graphs/compilation (set enforce_eager=True)")
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.enable_expert_parallel
        ):
            unsupported.append("parallel execution (the V2 adapter requires one GPU)")
        if parallel.enable_batch_sharded_sampling:
            unsupported.append("batch-sharded sampling")
        if config.speculative_config is not None or model.is_diffusion:
            unsupported.append("speculative/diffusion decoding")
        if (
            config.is_mm_encoder_only
            or model.is_multimodal_model
            or model.is_encoder_decoder
            or model.runner_type != "generate"
        ):
            unsupported.append("models other than text-only causal generation")
        if model.enable_trace_replay:
            unsupported.append("sampling trace replay")
        if config.cache_config.enable_prefix_caching:
            unsupported.append("prefix caching (set enable_prefix_caching=False)")
        if config.kv_transfer_config is not None:
            unsupported.append("KV transfer (cached activations bypass interventions)")
        if unsupported:
            raise ValueError(
                "NNsight Model Runner V2 does not support: " + "; ".join(unsupported)
            )

    def __init__(self, vllm_config, device):
        self.validate_config(vllm_config)
        super().__init__(vllm_config, device)

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        # The scoped sampling hook below relies on the ordinary Sampler contract.
        if type(self.sampler) is not Sampler or self.decode_query_len != 1:
            raise ValueError("NNsight Model Runner V2 does not support custom samplers")

    def _pause_requests(self):
        for mediator in self.nnsight_request_helper.mediators.values():
            mediator.active = False
            mediator.batch_group = None
        self.nnsight_model.interleaver.mediators = []

    def _set_batch_groups(self, input_batch, phase):
        self._pause_requests()
        interleaver = self.nnsight_model.interleaver
        if phase == "tokens":
            boundaries = input_batch.query_start_loc_np
            total = input_batch.num_tokens_after_padding
        elif phase == "logits":
            boundaries = input_batch.cu_num_logits_np
            total = int(boundaries[-1])
        elif phase == "samples":
            boundaries = np.arange(input_batch.num_reqs + 1)
            total = input_batch.num_reqs
        else:
            raise ValueError(f"Unknown intervention batch phase: {phase}")

        for row, req_id in enumerate(input_batch.req_ids):
            mediator = self.nnsight_request_helper.mediators.get(req_id)
            if mediator is None:
                continue
            start, end = int(boundaries[row]), int(boundaries[row + 1])
            mediator.batch_group = [start, end - start]
            mediator.active = True
            # Finished interventions may still own persistent cache hooks, but
            # putting them in this list would restart their worker threads.
            if mediator.alive:
                interleaver.mediators.append(mediator)

        # Batcher uses the full physical extent to recognize batch tensors.
        # Untraced requests and padding must count, even for a single mediator.
        interleaver.batcher.last_batch_group = [0, total]
        interleaver.batcher.needs_batching = True

    def add_requests(self, scheduler_output):
        super().add_requests(scheduler_output)
        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

    def prepare_inputs(self, *args, **kwargs):
        input_batch = super().prepare_inputs(*args, **kwargs)
        self._set_batch_groups(input_batch, "tokens")
        return input_batch

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
        context_len=0,
    ):
        self._pause_requests()
        result = None
        try:
            with self.nnsight_model.interleaver:
                result = super().execute_model(
                    scheduler_output,
                    intermediate_tensors,
                    dummy_run=dummy_run,
                    skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                    is_profile=is_profile,
                    context_len=context_len,
                )
        finally:
            self._pause_requests()
        # None is the normal V2 execute -> sample handoff. Do not manufacture
        # a ModelRunnerOutput here: that would bypass sampling in the engine.
        return result

    def _provide(self, name, value):
        interleaver = self.nnsight_model.interleaver
        mediators = interleaver.mediators
        try:
            for mediator in mediators:
                if not mediator.alive:
                    continue
                # Chain replacement tensors through each request. Calling the
                # existing fan-out with all mediators loses earlier replacements
                # when Batcher.swap returns a new tensor rather than editing it.
                interleaver.mediators = [mediator]
                prop = type(self.nnsight_model).__dict__[name]
                value = prop.provide(self.nnsight_model, value)
        finally:
            interleaver.mediators = [m for m in mediators if m.alive]
        return value

    @staticmethod
    def _validate_replacement(original, replacement, name):
        if (
            not isinstance(replacement, torch.Tensor)
            or replacement.shape != original.shape
            or replacement.dtype != original.dtype
            or replacement.device != original.device
        ):
            raise ValueError(
                f"V2 {name} interventions must preserve tensor shape, dtype, and device"
            )

    def sample(self, hidden_states, input_batch, grammar_output):
        self._set_batch_groups(input_batch, "logits")
        compute_logits = self.model.compute_logits
        sample = self.sampler.sample
        own_compute_logits = "compute_logits" in self.model.__dict__
        own_sample = "sample" in self.sampler.__dict__

        def provide_logits(*args, **kwargs):
            logits = compute_logits(*args, **kwargs)
            replacement = self._provide("logits", logits)
            self._validate_replacement(logits, replacement, "logits")
            return replacement

        def provide_samples(*args, **kwargs):
            sampled, processed_logits = sample(*args, **kwargs)
            self._set_batch_groups(input_batch, "samples")
            tokens = sampled.view(-1, 1)
            replacement = self._provide("samples", tokens)
            self._validate_replacement(tokens, replacement, "samples")
            # Sampler.__call__ computes logprobs from this returned token, then
            # sample_tokens copies it and updates GPU request state. All three
            # must consume the edited value, including out-of-place replacements.
            return replacement.reshape_as(sampled), processed_logits

        result = None
        try:
            self.model.compute_logits = provide_logits
            self.sampler.sample = provide_samples
            with self.nnsight_model.interleaver:
                result = super().sample(hidden_states, input_batch, grammar_output)
        finally:
            if own_compute_logits:
                self.model.compute_logits = compute_logits
            else:
                del self.model.compute_logits
            if own_sample:
                self.sampler.sample = sample
            else:
                del self.sampler.sample
            # Prompt-logprobs computation and profiling also call model modules;
            # neither should advance the request's intervention iteration.
            self._pause_requests()
        return result

    def shutdown(self):
        # Cancellation can happen without a final streamed output / collect RPC.
        self.nnsight_request_helper.discard_all()
        try:
            return super().shutdown()
        finally:
            self.nnsight_model = None

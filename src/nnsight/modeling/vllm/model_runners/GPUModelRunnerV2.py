"""Run interventions inside vLLM's second GPU model runner (Model Runner V2).

The tree, the request registry, what is served around the sampler and
collection are the V1 adapter's (`NNsightRunner`, `Requests`). What differs
is where vLLM's own path reaches the things a block reads:

* ``add_requests``: new requests hand over their workers, as ``_update_states``
  does on V1.
* ``prepare_inputs``: the batch order and each request's token count are
  settled here, inside ``execute_model``, so every worker's span is recomputed
  here.
* ``execute_model``: the forward, run with the interleaver open.
* ``sample``: the logits are computed, masked and sampled inside this one
  call. It runs with the interleaver open, so the modules ``compute_logits``
  calls hand off like any other, and the runner's sampler is swapped for
  `NNsightSampler`, which offers the logits and the sampled ids by location at
  the two points vLLM's sampler has them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler

from .GPUModelRunner import NNsightRunner


class NNsightSampler(Sampler):
    """vLLM's V2 sampler, offering its logits and its sampled ids to the workers.

    Swapped in for the class of the sampler the runner built, so it adds
    behaviour and no constructor state; ``nnsight_runner`` is set by the runner.
    """

    nnsight_runner: "NNsightGPUModelRunnerV2"

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> Any:
        # The runner has applied a structured-output grammar mask by now.
        return super().__call__(self.nnsight_runner._serve_logits(logits), input_batch)

    def sample(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        sampled, processed_logits = super().sample(*args, **kwargs)
        # Served as ``[requests, 1]``, the shape the V1 runner serves. A view, so
        # an in-place edit lands in `sampled`; a replacement comes back as a new
        # tensor, and it is what the logprobs and the request state are computed
        # from, so it takes `sampled`'s place.
        served = self.nnsight_runner._serve_samples(sampled.view(-1, 1))
        return served.reshape(sampled.shape), processed_logits


class NNsightGPUModelRunnerV2(NNsightRunner, GPUModelRunner):
    """vLLM's V2 model runner, interleaving interventions with the forward."""

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        # A full graph has no break points for a tap to be recorded at, and this
        # runner replays one (not a breakable graph, as V1 does) for every step
        # a full mode covers, where the taps would be silently skipped.
        mode = self.vllm_config.compilation_config.cudagraph_mode
        if self.nnsight_model.interleaver.taps and mode.has_full_cudagraphs():
            raise NotImplementedError(
                f"taps on vLLM's V2 model runner need piecewise CUDA graphs, but "
                f"this engine's cudagraph_mode is {mode.name}, whose full graphs "
                "replay without reaching a tap. Pass "
                "compilation_config={'cudagraph_mode': 'PIECEWISE'}, or set "
                "VLLM_USE_V2_MODEL_RUNNER=0 to use the V1 runner."
            )
        # None on a pipeline stage that does not sample, and for a pooling model.
        if self.sampler is not None:
            if type(self.sampler) is not Sampler:
                raise NotImplementedError(
                    f"nnsight serves logits and samples through vLLM's V2 Sampler, "
                    f"but this model installed {type(self.sampler).__name__} in its "
                    "place. Set VLLM_USE_V2_MODEL_RUNNER=0 to use the V1 runner."
                )
            self.sampler.__class__ = NNsightSampler
            self.sampler.nnsight_runner = self

    def add_requests(self, scheduler_output: Any) -> None:
        super().add_requests(scheduler_output)
        requests = self.nnsight_requests
        # Registered blocks are collected off the scheduler's own finished set:
        # a request nobody traced has no collect coming for it, and this is the
        # one place the runner is told it is over.
        if scheduler_output.finished_req_ids:
            requests.harvest(scheduler_output.finished_req_ids)
        requests.add(scheduler_output.scheduled_new_reqs, self.nnsight_persistent_objects)

    def prepare_inputs(self, *args: Any, **kwargs: Any) -> Any:
        input_batch = super().prepare_inputs(*args, **kwargs)

        # The batch in the order the forward's tensors follow, with each request's
        # token count for this step. Read off what vLLM returns rather than its
        # arguments, which differ between vLLM releases.
        spans = list(zip(input_batch.req_ids, input_batch.num_scheduled_tokens.tolist()))
        # How much of each prompt was computed before this step, in the shape
        # `refuse_chunked` reads off the V1 runner's request states.
        states = {
            request_id: SimpleNamespace(num_computed_tokens=int(computed), num_prompt_tokens=int(prompt))
            for request_id, computed, prompt in zip(
                input_batch.req_ids,
                input_batch.num_computed_prefill_tokens_np,
                input_batch.prefill_len_np,
            )
        }
        requests = self.nnsight_requests
        requests.refuse_chunked(spans, states)
        requests.scope(self.nnsight_model, spans)
        return input_batch

    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
        dummy_run: bool = False,
        **kwargs: Any,
    ) -> Any:
        # Warmup, profiling and graph capture run the forward with no requests
        # and never reach `prepare_inputs`; the interleaver is left as it is
        # (closed, or opened by `capture_model` to record the taps).
        if dummy_run:
            return super().execute_model(
                scheduler_output, intermediate_tensors, dummy_run=dummy_run, **kwargs
            )

        with self._open_step():
            output = super().execute_model(
                scheduler_output, intermediate_tensors, dummy_run=dummy_run, **kwargs
            )
            # The forward is done. What follows it (`compute_logits` on one row
            # per request, then the sampler) is per-request, not per-token.
            self.nnsight_requests.unflatten(self.nnsight_model)
        return output

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        # Re-entered for the per-request values (see `NNsightSampler`). A worker's
        # stop or error is deferred onto the worker, so nothing unwinds vLLM here.
        with self.nnsight_model.interleaver:
            return super().sample(*args, **kwargs)

"""CPU contracts for MRV2; set NNSIGHT_TEST_VLLM_V2=1 for GPU smoke tests.

The CPU tests run the real NNsight runner, batcher, and intervention threads.
Only the vLLM GPU boundary is replaced, so these tests do not require vLLM or
model downloads. The opt-in tests additionally exercise the installed engine.
"""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from nnsight.intervention.batching import Batcher
from nnsight.intervention.interleaver import (
    Cancelation,
    EarlyStopException,
    Interleaver,
    Mediator,
    eproperty,
)
from nnsight.intervention.tracing.globals import Globals


class _Mediator(Mediator):
    """Use actual event handling without a captured user Python frame."""

    def push(self):
        pass

    def pull(self):
        pass


class _Model:
    path = ""

    def __init__(self):
        self.interleaver = Interleaver(mediators=[], batcher=Batcher())

    @eproperty(iterate=True)
    def logits(self):
        pass

    @eproperty(iterate=True)
    def samples(self):
        pass

    def _remoteable_persistent_objects(self):
        return {}


class _Sampler:
    """MRV2's sample -> selected-token logprobs -> output boundary."""

    def sample(self, logits, *args, **kwargs):
        return logits.argmax(dim=-1), logits

    def __call__(self, logits, batch):
        sampled, processed = self.sample(logits)
        # A late intervention on SamplerOutput would leave these inconsistent.
        scores = processed.log_softmax(dim=-1).gather(1, sampled[:, None])
        return SimpleNamespace(
            sampled_token_ids=sampled[:, None],
            logprobs_tensors=scores,
            num_sampled=batch.num_sampled,
            num_rejected=torch.zeros_like(batch.num_sampled),
        )


class _GPUModelRunner:
    """Small upstream contract, including state feedback after sample()."""

    def add_requests(self, scheduler_output):
        pass

    def prepare_inputs(self, scheduler_output):
        return scheduler_output.input_batch

    def execute_model(self, scheduler_output, intermediate_tensors=None, **kwargs):
        self.add_requests(scheduler_output)
        batch = self.prepare_inputs(scheduler_output)
        self.forward_callback(batch)
        return None

    def sample(self, hidden_states, input_batch, grammar_output):
        logits = self.model.compute_logits(hidden_states[input_batch.logits_indices])
        if grammar_output is not None:
            grammar_output(logits)
        sampled = self.sampler(logits, input_batch)
        return sampled, sampled.num_sampled, sampled.num_rejected

    def sample_tokens(self, hidden_states, input_batch, grammar_output=None):
        output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
        self.copied_output = output.sampled_token_ids.clone()
        self.next_input_ids = output.sampled_token_ids.clone()
        return output, num_sampled, num_rejected


@pytest.fixture
def adapter(monkeypatch):
    """Load production code without importing vLLM's CUDA extension tree."""
    source = Path(__file__).resolve().parents[1] / "src/nnsight/modeling/vllm"
    package_name = "nnsight.modeling._vllm_v2_test"

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    module(package_name, __path__=[str(source)], VLLM=_Model)
    module(f"{package_name}.model_runners", __path__=[str(source / "model_runners")])
    module(f"{package_name}.batching", VLLMBatcher=Batcher)
    module("vllm", __path__=[], __version__="0.29.0")
    for name in (
        "distributed",
        "v1",
        "v1.worker",
        "v1.worker.gpu",
        "v1.worker.gpu.sample",
    ):
        module(f"vllm.{name}", __path__=[])
    module(
        "vllm.distributed.parallel_state",
        get_pp_group=lambda: SimpleNamespace(rank=0, world_size=1),
        get_tp_group=lambda: SimpleNamespace(rank=0, world_size=1),
    )
    module("vllm.tokenizers", cached_tokenizer_from_config=lambda config: None)
    module("vllm.v1.worker.gpu.model_runner", GPUModelRunner=_GPUModelRunner)
    module("vllm.v1.worker.gpu.sample.sampler", Sampler=_Sampler)

    def load(name):
        full_name = f"{package_name}.model_runners.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name, source / "model_runners" / f"{name}.py"
        )
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, loaded)
        spec.loader.exec_module(loaded)
        return loaded

    runtime = load("runtime")
    runner_module = load("GPUModelRunnerV2")
    return SimpleNamespace(
        runtime=runtime,
        module=runner_module,
        runner=runner_module.NNsightGPUModelRunnerV2,
    )


def _batch(req_ids, token_counts, *, logits_counts=None, padded_tokens=None):
    if logits_counts is None:
        logits_counts = [1] * len(req_ids)
    query_starts = np.cumsum([0, *token_counts], dtype=np.int32)
    logits_starts = np.cumsum([0, *logits_counts], dtype=np.int32)
    return SimpleNamespace(
        req_ids=req_ids,
        num_reqs=len(req_ids),
        num_reqs_after_padding=len(req_ids),
        num_tokens=int(query_starts[-1]),
        num_tokens_after_padding=padded_tokens or int(query_starts[-1]),
        query_start_loc_np=query_starts,
        cu_num_logits_np=logits_starts,
        logits_indices=torch.tensor(query_starts[1:] - 1, dtype=torch.long),
        num_logits=int(logits_starts[-1]),
        num_draft_tokens=0,
        num_sampled=torch.ones(len(req_ids), dtype=torch.int32),
        # Persistent slots deliberately differ from execution order.
        idx_mapping_np=np.arange(len(req_ids) - 1, -1, -1, dtype=np.int32),
    )


def _runner(adapter):
    runner = object.__new__(adapter.runner)
    runner.nnsight_model = _Model()
    runner.nnsight_request_helper = adapter.runtime.NNsightRequestHelper()
    runner.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    runner.sampler = _Sampler()
    return runner


def _start(runner, req_id, operation):
    def intervention(mediator, info):
        try:
            operation(mediator)
        except Cancelation as exc:
            mediator.exception(exc)
        except Exception as exc:
            mediator.exception(exc)
        else:
            mediator.end()

    mediator = _Mediator(intervention, info=SimpleNamespace(frame=None))
    interleaver = runner.nnsight_model.interleaver
    interleaver.mediators.append(mediator)
    runner.nnsight_request_helper.mediators[req_id] = mediator
    mediator.start(interleaver)
    return mediator


def _cancel(runner):
    for mediator in runner.nnsight_request_helper.mediators.values():
        mediator.cancel()
        mediator.remove_hooks()


def test_mixed_requests_and_padding_use_execution_order(adapter):
    runner = _runner(adapter)
    mediator = _start(runner, "traced", lambda m: m.request("unused"))
    batch = _batch(["ordinary-a", "traced", "ordinary-b"], [2, 3, 1], padded_tokens=8)
    try:
        runner._set_batch_groups(batch, "tokens")
        batcher = runner.nnsight_model.interleaver.batcher
        batcher.current_value = torch.arange(8)[:, None]
        assert mediator.batch_group == [2, 3]
        assert batcher.needs_batching
        assert batcher.narrow(mediator.batch_group).flatten().tolist() == [2, 3, 4]
        batcher.swap(mediator.batch_group, torch.full((3, 1), -1))
        assert batcher.current_value.flatten().tolist() == [0, 1, -1, -1, -1, 5, 6, 7]

        runner._set_batch_groups(batch, "logits")
        batcher.current_value = torch.arange(3)[:, None]
        assert batcher.narrow(mediator.batch_group).flatten().tolist() == [1]

        reordered = _batch(["traced", "ordinary-b"], [1, 2])
        runner._set_batch_groups(reordered, "tokens")
        assert mediator.batch_group == [0, 1]
    finally:
        _cancel(runner)


def test_unscheduled_request_loses_slice_without_restarting(adapter):
    runner = _runner(adapter)
    mediator = _start(runner, "paused", lambda m: m.request("samples.i0"))
    original_worker = mediator.worker
    try:
        runner._set_batch_groups(_batch(["paused", "ordinary"], [3, 1]), "tokens")
        runner._set_batch_groups(_batch(["ordinary"], [1]), "tokens")
        assert mediator.batch_group is None
        assert not mediator.active
        assert runner.nnsight_model.interleaver.mediators == []
        assert mediator.worker is original_worker

        runner._set_batch_groups(_batch(["ordinary", "paused"], [1, 1]), "samples")
        assert mediator.active
        assert mediator.batch_group == [1, 1]
        assert mediator.worker is original_worker
    finally:
        _cancel(runner)


def test_finished_intervention_is_not_restarted_on_next_step(adapter):
    runner = _runner(adapter)
    invocations = []
    mediator = _start(runner, "done", lambda m: invocations.append(True))
    assert not mediator.alive
    runner._set_batch_groups(_batch(["done"], [1]), "tokens")
    with runner.nnsight_model.interleaver:
        pass
    assert invocations == [True]
    assert not mediator.alive


def test_preempted_request_admission_is_idempotent(adapter, monkeypatch):
    runner = _runner(adapter)

    def intervention(mediator, info):
        try:
            mediator.request("samples.i0")
        except Cancelation as exc:
            mediator.exception(exc)

    mediator = _Mediator(intervention, info=SimpleNamespace(frame=None))
    deserialized = []

    def deserialize(payload, persistent_objects):
        deserialized.append(payload)
        return mediator

    monkeypatch.setattr(adapter.runtime, "load", deserialize)
    request = SimpleNamespace(
        req_id="request",
        sampling_params=SimpleNamespace(
            extra_args={
                "nnsight_trace_id": "trace",
                "nnsight_mediator": b"mediator",
                "nnsight_expected_count": 1,
            }
        ),
    )
    scheduler = SimpleNamespace(scheduled_new_reqs=[request])
    helper = runner.nnsight_request_helper
    try:
        runner.add_requests(scheduler)
        worker = mediator.worker
        runner._pause_requests()
        runner.add_requests(scheduler)
        assert deserialized == [b"mediator"]
        assert helper.mediators["request"] is mediator
        assert mediator.worker is worker
        assert helper.trace_contexts["trace"]["received_count"] == 1
        assert helper.trace_contexts["trace"]["pending_req_ids"] == {"request"}
    finally:
        _cancel(runner)


def test_execute_preserves_sampling_handoff_and_pauses_after_forward(adapter):
    runner = _runner(adapter)
    mediator = _start(runner, "traced", lambda m: m.request("samples.i0"))
    batch = _batch(["ordinary", "traced"], [2, 3])
    forwards = []

    def forward(input_batch):
        assert input_batch is batch
        assert runner.nnsight_model.interleaver.interleaving
        assert mediator.active
        assert mediator.batch_group == [2, 3]
        forwards.append(True)

    runner.forward_callback = forward
    scheduler = SimpleNamespace(scheduled_new_reqs=[], input_batch=batch)
    try:
        # None is required so the engine proceeds to sample_tokens().
        assert runner.execute_model(scheduler) is None
        assert forwards == [True]
        assert mediator.batch_group is None
        assert not mediator.active
        assert mediator.alive
    finally:
        _cancel(runner)


def test_replacement_samples_reach_logprobs_output_and_next_decode(adapter):
    runner = _runner(adapter)
    seen = []

    def intervene(mediator):
        seen.append(mediator.request("logits.i0").clone())
        mediator.swap("logits.i0", torch.tensor([[0.0, 3.0, 1.0]]))
        seen.append(mediator.request("samples.i0").clone())
        mediator.swap("samples.i0", torch.tensor([[2]]))

    _start(runner, "traced", intervene)
    batch = _batch(["ordinary-a", "traced", "ordinary-b"], [1, 1, 1])
    logits = torch.tensor([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    original_compute = runner.model.compute_logits
    original_sample = runner.sampler.sample
    try:
        runner._set_batch_groups(batch, "tokens")
        output, num_sampled, num_rejected = runner.sample_tokens(logits, batch)
        assert seen[0].tolist() == [[8.0, 0.0, 0.0]]
        assert seen[1].tolist() == [[1]]
        assert output.sampled_token_ids.flatten().tolist() == [0, 2, 1]
        assert runner.copied_output.flatten().tolist() == [0, 2, 1]
        assert runner.next_input_ids.flatten().tolist() == [0, 2, 1]
        expected_score = torch.tensor([0.0, 3.0, 1.0]).log_softmax(0)[2]
        torch.testing.assert_close(output.logprobs_tensors[1, 0], expected_score)
        assert num_sampled is batch.num_sampled
        assert num_rejected.tolist() == [0, 0, 0]
        assert runner.model.compute_logits == original_compute
        assert runner.sampler.sample == original_sample
    finally:
        _cancel(runner)


def test_multiple_replacement_interventions_accumulate(adapter):
    runner = _runner(adapter)

    def replace_with(token):
        def operation(mediator):
            mediator.request("samples.i0")
            mediator.swap("samples.i0", torch.tensor([[token]]))

        return operation

    _start(runner, "left", replace_with(1))
    _start(runner, "right", replace_with(2))
    batch = _batch(["left", "ordinary", "right"], [1, 1, 1])
    try:
        runner._set_batch_groups(batch, "samples")
        # A view forces Batcher.swap to allocate a replacement tensor.
        result = runner._provide("samples", torch.zeros(3, dtype=torch.long)[:, None])
        assert result.flatten().tolist() == [1, 0, 2]
    finally:
        _cancel(runner)


def test_grammar_runs_after_logits_intervention_and_wrappers_restore_on_error(adapter):
    runner = _runner(adapter)
    _start(runner, "traced", lambda m: m.swap("logits.i0", torch.tensor([[0.0, 9.0]])))
    batch = _batch(["traced"], [1])
    original_compute = runner.model.compute_logits
    original_sample = runner.sampler.sample

    def grammar(logits):
        assert logits.tolist() == [[0.0, 9.0]]
        raise RuntimeError("grammar failed")

    try:
        runner._set_batch_groups(batch, "tokens")
        with pytest.raises(RuntimeError, match="grammar failed"):
            runner.sample(torch.zeros((1, 2)), batch, grammar)
        assert runner.model.compute_logits == original_compute
        assert runner.sampler.sample == original_sample
    finally:
        _cancel(runner)


def test_nonfinal_prefill_keeps_upstream_zero_sample_count(adapter):
    runner = _runner(adapter)
    _start(runner, "prefill", lambda m: m.swap("samples.i0", torch.tensor([[1]])))
    batch = _batch(["prefill"], [2])
    batch.num_sampled.zero_()
    try:
        runner._set_batch_groups(batch, "tokens")
        output, num_sampled, _ = runner.sample(torch.zeros((2, 2)), batch, None)
        assert output.sampled_token_ids.tolist() == [[1]]
        assert num_sampled.tolist() == [0]
    finally:
        _cancel(runner)


@pytest.mark.parametrize(
    "exception", [ValueError("intervention failed"), EarlyStopException()]
)
def test_deferred_intervention_error_preserves_engine_sample(adapter, exception):
    runner = _runner(adapter)
    runner.nnsight_model.interleaver.defer_exceptions = True

    def operation(mediator):
        mediator.request("logits.i0")
        raise exception

    mediator = _start(runner, "traced", operation)
    # The worker does not need a trace frame to attach a deferred error.
    mediator.info = None
    batch = _batch(["traced"], [1])
    try:
        output, _, _ = runner.sample(torch.tensor([[0.0, 4.0]]), batch, None)
        assert output.sampled_token_ids.tolist() == [[1]]
        assert not mediator.alive
        assert mediator._deferred_type_name == type(exception).__name__
        assert mediator._deferred_is_control_flow == isinstance(
            exception, EarlyStopException
        )
        assert mediator.deferred_exception is not None
    finally:
        _cancel(runner)


def test_finalization_delivers_result_to_inactive_request(adapter):
    runner = _runner(adapter)
    results = []
    mediator = _start(
        runner, "request-internal", lambda m: results.append(m.request("result"))
    )
    module = torch.nn.Identity()
    mediator.hooks.append(module.register_forward_hook(lambda *args: None))
    runner._pause_requests()
    try:
        finished = runner.nnsight_request_helper.finalize_mediators(
            [("request", mediator, "request-internal")],
            {"request"},
            runner.nnsight_model,
        )
        assert results == [["request"]]
        assert finished == {"request-internal"}
        assert not mediator.alive
        assert not module._forward_hooks
    finally:
        _cancel(runner)


def test_discard_all_releases_hooks_and_uncollected_trace_saves(adapter, monkeypatch):
    runner = _runner(adapter)
    mediator = _start(runner, "request", lambda m: m.request("samples.i0"))
    local_save, shared_save, unrelated_save = [], [], []
    mediator.info.frame = SimpleNamespace(f_locals={"local": local_save})
    helper = runner.nnsight_request_helper
    helper.trace_contexts["incomplete"] = {
        "canonical_globals": {"shared": shared_save},
        "saved_names": ["shared"],
        "received_count": 1,
        "expected_count": 3,
        "pending_req_ids": {"request"},
    }
    monkeypatch.setattr(
        Globals, "saves", {id(local_save), id(shared_save), id(unrelated_save)}
    )
    module = torch.nn.Identity()
    mediator.hooks.append(module.register_forward_hook(lambda *args: None))
    try:
        helper.discard_all()
        assert not mediator.alive
        assert not module._forward_hooks
        assert helper.mediators == {}
        assert helper.trace_contexts == {}
        assert Globals.saves == {id(unrelated_save)}
        # Shutdown and explicit cancellation may both request cleanup.
        helper.discard_all()
    finally:
        _cancel(runner)


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=True,
            is_diffusion=False,
            is_multimodal_model=False,
            is_encoder_decoder=False,
            runner_type="generate",
            enable_trace_replay=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_batch_sharded_sampling=False,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
        kv_transfer_config=None,
        is_mm_encoder_only=False,
    )


@pytest.mark.parametrize("installed", ["0.29.0", "0.29.2"])
def test_supported_configuration_accepts_v029(adapter, monkeypatch, installed):
    monkeypatch.setattr(adapter.module, "version", lambda _: installed)
    adapter.runner.validate_config(_config())


@pytest.mark.parametrize("installed", ["0.28.0", "0.30.0"])
def test_unsupported_versions_fail_before_engine_initialization(
    adapter, monkeypatch, installed
):
    monkeypatch.setattr(adapter.module, "version", lambda _: installed)
    with pytest.raises(ValueError, match="requires vLLM"):
        adapter.runner.validate_config(_config())


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("model_config", "enforce_eager", False, "enforce_eager"),
        ("parallel_config", "tensor_parallel_size", 2, "one GPU"),
        ("parallel_config", "pipeline_parallel_size", 2, "one GPU"),
        ("parallel_config", "data_parallel_size", 2, "one GPU"),
        ("parallel_config", "decode_context_parallel_size", 2, "one GPU"),
        ("parallel_config", "prefill_context_parallel_size", 2, "one GPU"),
        ("parallel_config", "enable_expert_parallel", True, "one GPU"),
        ("parallel_config", "enable_batch_sharded_sampling", True, "batch-sharded"),
        ("model_config", "is_multimodal_model", True, "text-only"),
        ("model_config", "runner_type", "pooling", "text-only"),
        ("model_config", "enable_trace_replay", True, "trace replay"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching"),
        (None, "speculative_config", object(), "speculative"),
        (None, "kv_transfer_config", object(), "KV transfer"),
    ],
)
def test_unsupported_configuration_is_explicit(
    adapter, monkeypatch, section, field, value, message
):
    monkeypatch.setattr(adapter.module, "version", lambda _: "0.29.0")
    config = _config()
    target = config if section is None else vars(config)[section]
    setattr(target, field, value)
    with pytest.raises(ValueError, match=message):
        adapter.runner.validate_config(config)


@pytest.fixture(scope="module")
def gpu_v2_model():
    if os.environ.get("NNSIGHT_TEST_VLLM_V2") != "1":
        pytest.skip("Set NNSIGHT_TEST_VLLM_V2=1 to run the vLLM 0.29 GPU smoke tests")
    if not torch.cuda.is_available():
        pytest.skip("MRV2 integration tests require a CUDA GPU")
    pytest.importorskip("vllm")
    from nnsight.modeling.vllm import VLLM

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        model = VLLM(
            "openai-community/gpt2",
            dispatch=True,
            tensor_parallel_size=1,
            max_model_len=64,
            max_num_batched_tokens=16,
            gpu_memory_utilization=0.2,
            enable_prefix_caching=False,
        )
        try:
            yield model
        finally:
            model.vllm_entrypoint.llm_engine.engine_core.shutdown()


@torch.no_grad()
def test_gpu_forced_sample_is_returned_and_fed_into_next_decode(
    gpu_v2_model, monkeypatch
):
    model = gpu_v2_model
    forced_id = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
    outputs = []
    generate = model.vllm_entrypoint.generate

    def capture(*args, **kwargs):
        result = generate(*args, **kwargs)
        outputs.extend(result)
        return result

    monkeypatch.setattr(model.vllm_entrypoint, "generate", capture)
    with model.trace(
        "Hello", max_tokens=2, temperature=0.0, ignore_eos=True, logprobs=1
    ) as tracer:
        with tracer.iter[0]:
            first_logits = model.logits.clone().save()
            model.samples = torch.full_like(model.samples, forced_id)
        with tracer.iter[1]:
            next_input = model.inputs[1]["input_ids"].clone().save()

    assert outputs[0].outputs[0].token_ids[0] == forced_id
    assert next_input.flatten().tolist() == [forced_id]
    expected_logprob = first_logits.float().log_softmax(-1)[0, forced_id].item()
    assert outputs[0].outputs[0].logprobs[0][forced_id].logprob == pytest.approx(
        expected_logprob, abs=1e-4
    )


@torch.no_grad()
def test_gpu_logits_replacement_controls_sampling(gpu_v2_model):
    model = gpu_v2_model
    forced_id = model.tokenizer.encode(" Tokyo", add_special_tokens=False)[0]
    with model.trace("Hello", max_tokens=1, temperature=0.0, ignore_eos=True):
        replacement = torch.full_like(model.logits, float("-inf"))
        replacement[:, forced_id] = 0
        model.logits = replacement
        sampled = model.samples.clone().save()
    assert sampled.item() == forced_id


@torch.no_grad()
def test_gpu_chunked_prefill_counts_executions(gpu_v2_model):
    model = gpu_v2_model
    prompt = model.tokenizer.encode("hello " * 35, add_special_tokens=False)
    assert 16 < len(prompt) < 64
    with model.trace(prompt, max_tokens=1, temperature=0.0) as tracer:
        chunk_sizes = [].save()
        candidates = [].save()
        for step in tracer.iter[:]:
            chunk_sizes.append(model.transformer.h[0].mlp.output.shape[0])
            candidates.append(model.samples.item())
    assert sum(chunk_sizes) == len(prompt)
    assert max(chunk_sizes) <= 16
    assert len(chunk_sizes) == len(candidates) > 1


@torch.no_grad()
def test_gpu_mixed_requests_keep_interventions_separate(gpu_v2_model, monkeypatch):
    from vllm import SamplingParams

    model = gpu_v2_model
    paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
    tokyo = model.tokenizer.encode(" Tokyo", add_special_tokens=False)[0]
    generate = model.vllm_entrypoint.generate
    outputs = []

    def add_untraced(prompts, sampling_params, lora_request):
        ordinary = SamplingParams(max_tokens=1, temperature=0.0)
        result = generate(
            ["Untraced before", *prompts, "Untraced after"],
            sampling_params=[ordinary, *sampling_params, ordinary],
            lora_request=[None, *lora_request, None],
        )
        outputs.extend(result)
        return result

    monkeypatch.setattr(model.vllm_entrypoint, "generate", add_untraced)
    with model.trace(max_tokens=1, temperature=0.0) as tracer:
        with tracer.invoke("Hello"):
            model.samples = torch.full_like(model.samples, paris)
        with tracer.invoke("Good day"):
            model.samples = torch.full_like(model.samples, tokyo)
    assert outputs[1].outputs[0].token_ids == [paris]
    assert outputs[2].outputs[0].token_ids == [tokyo]

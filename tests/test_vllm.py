"""
Tests for vLLM integration with nnsight.

These tests cover vLLM-specific features:
- Basic inference and logit access
- Multi-token generation
- Sampling with different parameters
- Activation interventions
- Batched operations
- Tensor parallelism
- Async engine with streaming saves
"""

import asyncio

import pytest
import torch
from typing import TYPE_CHECKING

try:
    from nnsight.modeling.vllm import VLLM
except Exception as e:
    pytest.skip(f"Skipping VLLM tests: \n{e}", allow_module_level=True)

try:
    import ray
    _has_ray = True
except ImportError:
    _has_ray = False

_ray_skip = pytest.mark.skipif(
    not _has_ray or torch.cuda.device_count() < 2,
    reason="Ray tests require ray package and at least 2 GPUs",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tp(request):
    """Get tensor parallel size from command line."""
    tp = request.config.getoption("--tp")
    if tp > torch.cuda.device_count() or tp < 1:
        pytest.exit("--tp can't be higher than the number of available GPUs.")
    return tp


@pytest.fixture(scope="module")
def vllm_gpt2(tp: int):
    """Load GPT-2 model with vLLM."""
    return VLLM(
        "gpt2", tensor_parallel_size=tp, gpu_memory_utilization=0.1, dispatch=True
    )


# =============================================================================
# Basic Inference
# =============================================================================


class TestBasicInference:
    """Tests for basic vLLM inference."""

    @torch.no_grad()
    def test_single_logit(self, vllm_gpt2, ET_prompt: str):
        """Test single token logit prediction."""
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.output.save()

        next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
        assert next_token == " Paris"

    @torch.no_grad()
    def test_request_cleanup(self, vllm_gpt2, ET_prompt: str, MSG_prompt: str):
        """Test that requests are properly cleaned up between traces."""
        with vllm_gpt2.trace() as tracer:
            with tracer.invoke(ET_prompt):
                pass
            with tracer.invoke(MSG_prompt):
                pass

        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1, max_tokens=3
        ) as tracer:
            logits = list().save()
            with tracer.iter[0:3]:
                logits.append(vllm_gpt2.logits.output)

        assert vllm_gpt2.tokenizer.batch_decode(
            [logit.argmax(dim=-1) for logit in logits]
        ) == [" New", " York", " City"]


# =============================================================================
# Multi-Token Generation
# =============================================================================


class TestGeneration:
    """Tests for multi-token generation."""

    @torch.no_grad()
    def test_multi_token_generation(self, vllm_gpt2, MSG_prompt: str):
        """Test generating multiple tokens."""
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=3
        ) as tracer:
            logits = list().save()
            with tracer.iter[0:3]:
                logits.append(vllm_gpt2.logits.output)

        assert vllm_gpt2.tokenizer.batch_decode(
            [logit.argmax(dim=-1) for logit in logits]
        ) == [" New", " York", " City"]

    @torch.no_grad()
    def test_max_token_generation(self, vllm_gpt2, ET_prompt: str):
        """Test max token generation."""
        with vllm_gpt2.trace(ET_prompt, max_tokens=10) as tracer:
            logits = list().save()
            with tracer.all():
                logits.append(vllm_gpt2.logits.output)

        assert len(logits) == 10


# =============================================================================
# Sampling
# =============================================================================


class TestSampling:
    """Tests for sampling with different parameters."""

    @torch.no_grad()
    def test_sampling_temperature(self, vllm_gpt2, MSG_prompt: str):
        """Test sampling with different temperatures."""
        with vllm_gpt2.trace(max_tokens=3) as tracer:
            with tracer.invoke(MSG_prompt, temperature=0.8, top_p=0.95):
                samples_2 = list().save()
                with tracer.iter[0:3]:
                    samples_2.append(vllm_gpt2.samples.output.item())

            with tracer.invoke(MSG_prompt, temperature=0.0, top_p=1.0):
                samples_1 = list().save()
                with tracer.iter[0:3]:
                    samples_1.append(vllm_gpt2.samples.output.item())

        assert vllm_gpt2.tokenizer.batch_decode(
            samples_1
        ) != vllm_gpt2.tokenizer.batch_decode(samples_2)


# =============================================================================
# Interventions
# =============================================================================


class TestInterventions:
    """Tests for activation interventions."""

    @torch.no_grad()
    def test_basic_intervention(self, vllm_gpt2, ET_prompt: str):
        """Test basic activation intervention."""
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1) as tracer:
            out = vllm_gpt2.transformer.h[-2].mlp.output.clone()
            out[:] = 0
            vllm_gpt2.transformer.h[-2].mlp.output = out
            hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
            logits = vllm_gpt2.logits.output.save()

        next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
        assert torch.all(hs == 0)
        assert next_token == " London"

    @torch.no_grad()
    def test_swap_intervention(self, vllm_gpt2, ET_prompt: str):
        """Test swap-style intervention."""
        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1) as tracer:
            vllm_gpt2.transformer.h[-2].mlp.output = torch.zeros_like(
                vllm_gpt2.transformer.h[-2].mlp.output
            )
            hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
            logits = vllm_gpt2.logits.output.save()

        next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
        assert next_token == " London"
        assert torch.all(hs == 0)

    @torch.no_grad()
    def test_generation_with_intervention(self, tp, vllm_gpt2, MSG_prompt: str):
        """Test intervention during multi-token generation."""
        with vllm_gpt2.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=5
        ) as tracer:
            logits = list().save()
            hs_list = list().save()
            with tracer.iter[:] as it:
                if it == 2:
                    out = vllm_gpt2.transformer.h[-2].output.clone()
                    out[0][:] = 0
                    vllm_gpt2.transformer.h[-2].output = out

                hs_list.append(vllm_gpt2.transformer.h[-2].output[0])
                logits.append(vllm_gpt2.logits.output)

        assert [torch.all(hs == 0) for hs in hs_list] == [
            False,
            False,
            True,
            False,
            False,
        ]

        if tp == 1:
            assert vllm_gpt2.tokenizer.decode(logits[2].argmax(dim=-1)) != " City"


# =============================================================================
# Batching
# =============================================================================


class TestBatching:
    """Tests for batched operations."""

    @torch.no_grad()
    def test_batched_intervention(self, vllm_gpt2, ET_prompt: str):
        """Test intervention on batched inputs."""
        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                clean_hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
                clean_logits = vllm_gpt2.logits.output.save()

            with tracer.invoke(ET_prompt):
                out = vllm_gpt2.transformer.h[-2].mlp.output[:].clone()
                out[:] = 0
                vllm_gpt2.transformer.h[-2].mlp.output = out
                corrupted_hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
                corrupted_logits = vllm_gpt2.logits.output.save()

        clean_token = vllm_gpt2.tokenizer.decode(clean_logits.argmax(dim=-1))
        corrupted_token = vllm_gpt2.tokenizer.decode(corrupted_logits.argmax(dim=-1))

        assert not torch.all(clean_hs == 0)
        assert torch.all(corrupted_hs == 0)
        assert clean_token == " Paris"
        assert corrupted_token == " London"

    @torch.no_grad()
    def test_batched_multi_token_with_iter(
        self, vllm_gpt2, ET_prompt: str, MSG_prompt: str
    ):
        """Test batched generation with different iteration ranges."""
        with vllm_gpt2.trace(max_tokens=10) as tracer:
            with tracer.invoke(ET_prompt):
                ET_logits = list().save()
                with tracer.iter[1:7]:
                    ET_logits.append(vllm_gpt2.logits.output)

            with tracer.invoke(MSG_prompt, max_tokens=5):
                MSG_logits = list().save()
                with tracer.iter[:5]:
                    MSG_logits.append(vllm_gpt2.logits.output)

        assert len(ET_logits) == 6
        assert len(MSG_logits) == 5


# =============================================================================
# Tensor Parallelism
# =============================================================================


class TestTensorParallelism:
    """Tests for tensor parallelism support."""

    @torch.no_grad()
    def test_tensor_parallelism(self, tp, vllm_gpt2, ET_prompt: str):
        """Test intervention with tensor parallelism."""
        if tp < 2:
            pytest.skip("Skipping test for tp>1!")

        with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1.0):
            value_tuple = vllm_gpt2.transformer.h[5].mlp.c_fc.output
            value = value_tuple[0].clone()
            value[:, 2000:] = 0
            value_tuple = (value, *value_tuple[1:])
            vllm_gpt2.transformer.h[5].mlp.c_fc.output = value_tuple
            hs = vllm_gpt2.transformer.h[5].mlp.c_fc.output[0].save()
            logit = vllm_gpt2.logits.output.save()
            next_token = vllm_gpt2.tokenizer.decode(logit.argmax(dim=-1)).save()

        assert next_token != " Paris"
        assert hs.shape == torch.Size([11, 3072])
        assert torch.all(hs[:, 2000:] == 0)


# =============================================================================
# Token Input Compatibility
# =============================================================================


class TestTokenInputs:
    """Tests for token ID and HuggingFace tokenizer input compatibility."""

    @torch.no_grad()
    def test_single_token_list(self, vllm_gpt2, ET_prompt: str):
        """Test passing a single list of token IDs."""
        token_ids = vllm_gpt2.tokenizer.encode(ET_prompt)

        with vllm_gpt2.trace(token_ids, temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.output.save()

        next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
        assert next_token == " Paris"

    @torch.no_grad()
    def test_hf_tokenizer_dict_single(self, vllm_gpt2, ET_prompt: str):
        """Test passing HuggingFace tokenizer output dict for single prompt."""
        hf_output = vllm_gpt2.tokenizer(ET_prompt, return_tensors="pt")

        with vllm_gpt2.trace(dict(hf_output), temperature=0.0, top_p=1):
            logits = vllm_gpt2.logits.output.save()

        next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
        assert next_token == " Paris"

    @torch.no_grad()
    def test_token_list_in_invoker(self, vllm_gpt2, ET_prompt: str):
        """Test token list input within an invoker."""
        token_ids = vllm_gpt2.tokenizer.encode(ET_prompt)

        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(token_ids):
                logits = vllm_gpt2.logits.output.save()

        next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
        assert next_token == " Paris"

    @torch.no_grad()
    def test_mixed_string_and_token_invokers(
        self, vllm_gpt2, ET_prompt: str, MSG_prompt: str
    ):
        """Test mixing string and token list inputs across invokers."""
        et_tokens = vllm_gpt2.tokenizer.encode(ET_prompt)

        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(et_tokens):
                et_logits = vllm_gpt2.logits.output.save()

            with tracer.invoke(MSG_prompt):
                msg_logits = vllm_gpt2.logits.output.save()

        et_token = vllm_gpt2.tokenizer.decode(et_logits.argmax(dim=-1))
        msg_token = vllm_gpt2.tokenizer.decode(msg_logits.argmax(dim=-1))
        assert et_token == " Paris"
        assert msg_token == " New"


# =============================================================================
# Ray Distributed Executor
# =============================================================================


@pytest.fixture(scope="module")
def vllm_gpt2_ray():
    """Load GPT-2 model with vLLM using Ray distributed executor."""
    if not _has_ray:
        pytest.skip("Ray not installed")
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for Ray executor")
    return VLLM(
        "gpt2",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.1,
        dispatch=True,
    )


@pytest.mark.skipif(
    not _has_ray or torch.cuda.device_count() < 2,
    reason="Ray tests require ray package and at least 2 GPUs",
)
class TestRayExecutor:
    """Tests for Ray distributed executor backend.

    Validates that NNsight interventions work correctly when vLLM uses
    RayDistributedExecutor instead of MultiprocExecutor. Requires the
    LazyRayWorkerWrapper patch (applied automatically by VLLM._load()).
    """

    @torch.no_grad()
    def test_ray_basic_logit(self, vllm_gpt2_ray, ET_prompt: str):
        """Test basic logit access with Ray executor."""
        with vllm_gpt2_ray.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_gpt2_ray.logits.output.save()

        next_token = vllm_gpt2_ray.tokenizer.decode(logits.argmax(dim=-1))
        assert next_token == " Paris"

    @torch.no_grad()
    def test_ray_intervention(self, vllm_gpt2_ray, ET_prompt: str):
        """Test activation intervention with Ray executor."""
        with vllm_gpt2_ray.trace(ET_prompt, temperature=0.0, top_p=1):
            out = vllm_gpt2_ray.transformer.h[-2].mlp.output.clone()
            out[:] = 0
            vllm_gpt2_ray.transformer.h[-2].mlp.output = out
            hs = vllm_gpt2_ray.transformer.h[-2].mlp.output.save()
            logits = vllm_gpt2_ray.logits.output.save()

        assert torch.all(hs == 0)

    @torch.no_grad()
    def test_ray_multi_token_generation(self, vllm_gpt2_ray, MSG_prompt: str):
        """Test multi-token generation with Ray executor."""
        with vllm_gpt2_ray.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=3
        ) as tracer:
            logits = list().save()
            with tracer.iter[0:3]:
                logits.append(vllm_gpt2_ray.logits.output)

        assert len(logits) == 3

    @torch.no_grad()
    def test_ray_generation_with_intervention(self, vllm_gpt2_ray, MSG_prompt: str):
        """Test intervention during multi-token generation with Ray executor."""
        with vllm_gpt2_ray.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=5
        ) as tracer:
            logits = list().save()
            hs_list = list().save()
            with tracer.iter[:] as it:
                if it == 2:
                    out = vllm_gpt2_ray.transformer.h[-2].output.clone()
                    out[0][:] = 0
                    vllm_gpt2_ray.transformer.h[-2].output = out

                hs_list.append(vllm_gpt2_ray.transformer.h[-2].output[0])
                logits.append(vllm_gpt2_ray.logits.output)

        assert [torch.all(hs == 0) for hs in hs_list] == [
            False,
            False,
            True,
            False,
            False,
        ]


# =============================================================================
# Cross-Invoke Shared State
# =============================================================================


class TestCrossInvokeSharedState:
    """Tests for shared state across invokes (e.g., shared saved lists)."""

    @torch.no_grad()
    def test_shared_list_across_invokes(self, vllm_gpt2, ET_prompt: str, MSG_prompt: str):
        """Test that a shared saved list collects values from multiple invokes."""
        prompts = [ET_prompt, MSG_prompt]

        with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:
            out_ids = [list() for _ in range(len(prompts))].save()
            for i, prompt in enumerate(prompts):
                with tracer.invoke(prompt):
                    out_ids[i].append(vllm_gpt2.logits.output.argmax(dim=-1))

        # Each sub-list should have exactly one entry (single-token generation)
        assert len(out_ids) == 2
        assert len(out_ids[0]) == 1
        assert len(out_ids[1]) == 1

        # Verify the predictions are correct
        et_token = vllm_gpt2.tokenizer.decode(out_ids[0][0])
        msg_token = vllm_gpt2.tokenizer.decode(out_ids[1][0])
        assert et_token == " Paris"
        assert msg_token == " New"

    @torch.no_grad()
    def test_shared_list_multi_token(self, vllm_gpt2, ET_prompt: str, MSG_prompt: str):
        """Test shared list with multi-token generation using tracer.all()."""
        prompts = [ET_prompt, MSG_prompt]
        num_tokens = 3

        with vllm_gpt2.trace(temperature=0.0, top_p=1, max_tokens=num_tokens) as tracer:
            out_ids = [list() for _ in range(len(prompts))].save()
            for i, prompt in enumerate(prompts):
                with tracer.invoke(prompt):
                    with tracer.all():
                        out_ids[i].append(vllm_gpt2.logits.output.argmax(dim=-1))

        # Each sub-list should have num_tokens entries
        assert len(out_ids) == 2
        assert len(out_ids[0]) == num_tokens
        assert len(out_ids[1]) == num_tokens


# =============================================================================
# Async Engine
# =============================================================================


@pytest.fixture(scope="module")
def async_loop():
    """Create a shared event loop for all async tests.

    AsyncLLM binds its background tasks to the event loop that created
    it, so all tests must share the same loop.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def vllm_gpt2_async(tp: int, async_loop):
    """Load GPT-2 model with vLLM async engine."""
    return VLLM(
        "gpt2",
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.1,
        dispatch=True,
        mode="async",
    )


class TestAsyncEngine:
    """Tests for async vLLM engine with streaming saves."""

    def test_async_basic_streaming(
        self, vllm_gpt2_async, async_loop, ET_prompt: str
    ):
        """Test that async engine streams multiple outputs."""

        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                logits = vllm_gpt2_async.logits.output.save()

            count = 0
            async for output in tracer.backend():
                count += 1

            assert count > 1, f"Expected streaming outputs, got {count}"

        async_loop.run_until_complete(run())

    def test_async_saves_on_every_output(
        self, vllm_gpt2_async, async_loop, ET_prompt: str
    ):
        """Test that saves are attached to every streamed output, not just the final one."""

        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                logits = vllm_gpt2_async.logits.output.save()

            count = 0
            saves_count = 0
            async for output in tracer.backend():
                count += 1
                if hasattr(output, "saves") and output.saves:
                    saves_count += 1
                    assert "logits" in output.saves
                    assert output.saves["logits"].shape[-1] == 50257

            assert saves_count == count, (
                f"Expected saves on every output, got {saves_count}/{count}"
            )

        async_loop.run_until_complete(run())

    def test_async_finished_flag(
        self, vllm_gpt2_async, async_loop, ET_prompt: str
    ):
        """Test that exactly the last output has finished=True."""

        async def run():
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=3
            ) as tracer:
                logits = vllm_gpt2_async.logits.output.save()

            outputs = []
            async for output in tracer.backend():
                outputs.append(output)

            assert len(outputs) >= 1
            for o in outputs[:-1]:
                assert not o.finished
            assert outputs[-1].finished

        async_loop.run_until_complete(run())

    def test_async_intervention(
        self, vllm_gpt2_async, async_loop, ET_prompt: str
    ):
        """Test activation intervention through the async path."""

        async def run():
            # Clean run (no intervention)
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=1
            ) as tracer:
                logits = vllm_gpt2_async.logits.output.save()

            clean_saves = None
            async for output in tracer.backend():
                if hasattr(output, "saves") and output.saves:
                    clean_saves = output.saves

            assert clean_saves is not None
            clean_token = vllm_gpt2_async.tokenizer.decode(
                clean_saves["logits"].argmax(dim=-1)
            )
            assert clean_token == " Paris"

            # Corrupted run (zero out MLP)
            with vllm_gpt2_async.trace(
                ET_prompt, temperature=0.0, max_tokens=1
            ) as tracer:
                vllm_gpt2_async.transformer.h[-2].mlp.output = torch.zeros_like(
                    vllm_gpt2_async.transformer.h[-2].mlp.output
                )
                logits = vllm_gpt2_async.logits.output.save()

            corrupted_saves = None
            async for output in tracer.backend():
                if hasattr(output, "saves") and output.saves:
                    corrupted_saves = output.saves

            assert corrupted_saves is not None
            corrupted_token = vllm_gpt2_async.tokenizer.decode(
                corrupted_saves["logits"].argmax(dim=-1)
            )
            assert corrupted_token != " Paris"

        async_loop.run_until_complete(run())

    def test_async_text_output(
        self, vllm_gpt2_async, async_loop, MSG_prompt: str
    ):
        """Test that streamed text output is non-empty."""

        async def run():
            with vllm_gpt2_async.trace(
                MSG_prompt, temperature=0.0, max_tokens=3
            ) as tracer:
                logits = vllm_gpt2_async.logits.output.save()

            texts = []
            async for output in tracer.backend():
                if output.outputs:
                    texts.append(output.outputs[0].text)

            assert len(texts) > 0
            # Final text should have content
            assert len(texts[-1].strip()) > 0

        async_loop.run_until_complete(run())


# =============================================================================
# Async Engine + Ray Distributed Executor
# =============================================================================


@pytest.fixture(scope="module")
def async_ray_loop():
    """Create a separate event loop for async + Ray tests.

    AsyncLLM binds to the loop that created it, so the async Ray model
    needs its own loop (distinct from the plain async model's loop).
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def vllm_gpt2_async_ray(async_ray_loop):
    """Load GPT-2 model with async vLLM engine + Ray distributed executor."""
    if not _has_ray:
        pytest.skip("Ray not installed")
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for Ray executor")
    return VLLM(
        "gpt2",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.1,
        dispatch=True,
        mode="async",
    )


@pytest.mark.skipif(
    not _has_ray or torch.cuda.device_count() < 2,
    reason="Async Ray tests require ray package and at least 2 GPUs",
)
class TestAsyncRayExecutor:
    """Tests for async vLLM engine with Ray distributed executor.

    Validates that async streaming + NNsight interventions work correctly
    when vLLM uses the NNsightRayExecutor backend.
    """

    def test_async_ray_basic_streaming(
        self, vllm_gpt2_async_ray, async_ray_loop, ET_prompt: str
    ):
        """Test that async + Ray engine streams multiple outputs."""

        async def run():
            with vllm_gpt2_async_ray.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                logits = vllm_gpt2_async_ray.logits.output.save()

            count = 0
            async for output in tracer.backend():
                count += 1

            assert count > 1, f"Expected streaming outputs, got {count}"

        async_ray_loop.run_until_complete(run())

    def test_async_ray_saves_on_every_output(
        self, vllm_gpt2_async_ray, async_ray_loop, ET_prompt: str
    ):
        """Test saves attached to every streamed output with Ray backend."""

        async def run():
            with vllm_gpt2_async_ray.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                logits = vllm_gpt2_async_ray.logits.output.save()

            count = 0
            saves_count = 0
            async for output in tracer.backend():
                count += 1
                if hasattr(output, "saves") and output.saves:
                    saves_count += 1
                    assert "logits" in output.saves
                    assert output.saves["logits"].shape[-1] == 50257

            assert saves_count == count, (
                f"Expected saves on every output, got {saves_count}/{count}"
            )

        async_ray_loop.run_until_complete(run())

    def test_async_ray_intervention(
        self, vllm_gpt2_async_ray, async_ray_loop, ET_prompt: str
    ):
        """Test activation intervention through async + Ray path."""

        async def run():
            # Clean run (no intervention)
            with vllm_gpt2_async_ray.trace(
                ET_prompt, temperature=0.0, max_tokens=1
            ) as tracer:
                logits = vllm_gpt2_async_ray.logits.output.save()

            clean_saves = None
            async for output in tracer.backend():
                if hasattr(output, "saves") and output.saves:
                    clean_saves = output.saves

            assert clean_saves is not None
            clean_token = vllm_gpt2_async_ray.tokenizer.decode(
                clean_saves["logits"].argmax(dim=-1)
            )
            assert clean_token == " Paris"

            # Corrupted run (zero out MLP)
            with vllm_gpt2_async_ray.trace(
                ET_prompt, temperature=0.0, max_tokens=1
            ) as tracer:
                vllm_gpt2_async_ray.transformer.h[-2].mlp.output = torch.zeros_like(
                    vllm_gpt2_async_ray.transformer.h[-2].mlp.output
                )
                logits = vllm_gpt2_async_ray.logits.output.save()

            corrupted_saves = None
            async for output in tracer.backend():
                if hasattr(output, "saves") and output.saves:
                    corrupted_saves = output.saves

            assert corrupted_saves is not None
            corrupted_token = vllm_gpt2_async_ray.tokenizer.decode(
                corrupted_saves["logits"].argmax(dim=-1)
            )
            assert corrupted_token != " Paris"

        async_ray_loop.run_until_complete(run())

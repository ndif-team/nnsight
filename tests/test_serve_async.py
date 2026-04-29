"""Thorough async/non-blocking tests for nnsight-serve.

Tests:
1. Timing overlap — are requests truly concurrent?
2. Result isolation — same variable names across traces don't interfere
3. Multi-process clients — separate processes hitting the same server
4. Dependent requests — what happens (should not corrupt engine)

Start the server first:
    CUDA_VISIBLE_DEVICES=1 conda run -n ndif-dev --no-capture-output bash -c \
        "python -m nnsight.modeling.vllm.serve.cli Qwen/Qwen2.5-0.5B-Instruct --port 6679 --gpu-memory-utilization 0.3"
"""

import os
import sys
import time
import multiprocessing as mp

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import pytest
import torch

try:
    from nnsight.modeling.vllm import VLLM
except Exception as e:
    pytest.skip(f"vllm import failed: {e}", allow_module_level=True)

import httpx

SERVE_URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6679")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ET_PROMPT = "The Eiffel Tower is located in the city of"
MSG_PROMPT = "Madison Square Garden is located in the city of"

if not httpx.get(f"{SERVE_URL}/health", timeout=5.0).status_code == 200:
    pytest.skip(f"Server not reachable at {SERVE_URL}", allow_module_level=True)


@pytest.fixture(scope="module")
def model():
    m = VLLM(MODEL)
    assert not m.dispatched
    return m


# =========================================================================
# 1. Timing overlap: are non-blocking requests truly concurrent?
# =========================================================================

class TestTimingOverlap:
    """Verify that non-blocking requests overlap in time."""

    def test_concurrent_faster_than_sequential(self, model):
        """4 non-blocking requests should take less than 4 * single_request_time.

        If they were truly sequential, total time ≈ N * single.
        If concurrent, total time ≈ single (+ batching overhead).
        We use a loose check: concurrent < 0.8 * sequential estimate.
        """
        # Warmup
        with model.trace(ET_PROMPT, serve=SERVE_URL):
            model.logits.output.save()

        # Measure single request time
        t0 = time.perf_counter()
        with model.trace(ET_PROMPT, serve=SERVE_URL):
            model.logits.output.save()
        single_time = time.perf_counter() - t0

        # Fire 4 non-blocking requests concurrently.
        # Each must use a proper `with` block (nnsight uses AST extraction).
        t0 = time.perf_counter()

        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t1:
            model.logits.output.save()
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t2:
            model.logits.output.save()
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t3:
            model.logits.output.save()
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t4:
            model.logits.output.save()

        # Wait for all to complete
        t1.collect(timeout=60)
        t2.collect(timeout=60)
        t3.collect(timeout=60)
        t4.collect(timeout=60)
        concurrent_time = time.perf_counter() - t0

        N = 4
        sequential_estimate = single_time * N
        print(f"\nSingle: {single_time:.3f}s, Concurrent({N}): {concurrent_time:.3f}s, "
              f"Sequential estimate: {sequential_estimate:.3f}s")

        # Concurrent should be meaningfully faster than sequential
        assert concurrent_time < sequential_estimate * 0.8, (
            f"Concurrent ({concurrent_time:.3f}s) not faster than "
            f"80% of sequential estimate ({sequential_estimate:.3f}s). "
            f"Requests may not be truly concurrent."
        )


# =========================================================================
# 2. Result isolation: same variable names across traces
# =========================================================================

class TestResultIsolation:
    """Verify that concurrent traces with same variable names don't interfere."""

    def test_same_varname_different_prompts(self, model):
        """Two traces both save as 'logits' — each should get its own result."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t1:
            logits = model.logits.output.save()

        with model.trace(MSG_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t2:
            logits = model.logits.output.save()

        saves1 = t1.collect(timeout=30)
        saves2 = t2.collect(timeout=30)

        # Both should have "logits" key
        assert "logits" in saves1, f"saves1 keys: {list(saves1.keys())}"
        assert "logits" in saves2, f"saves2 keys: {list(saves2.keys())}"

        # But the values should be different (different prompts)
        assert not torch.equal(saves1["logits"], saves2["logits"]), (
            "Same variable name from different prompts returned identical tensors — "
            "results are likely cross-contaminated"
        )

        # Verify correctness
        et_token = model.tokenizer.decode(saves1["logits"].argmax(dim=-1))
        msg_token = model.tokenizer.decode(saves2["logits"].argmax(dim=-1))
        assert "Paris" in et_token, f"Expected 'Paris', got '{et_token}'"
        assert "New" in msg_token, f"Expected 'New', got '{msg_token}'"

    def test_same_varname_same_prompt(self, model):
        """Two traces, same prompt, same varname — results should be identical."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t1:
            logits = model.logits.output.save()

        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t2:
            logits = model.logits.output.save()

        saves1 = t1.collect(timeout=30)
        saves2 = t2.collect(timeout=30)

        assert torch.equal(saves1["logits"], saves2["logits"]), (
            f"Same prompt, same varname but different results. "
            f"Max diff: {(saves1['logits'].float() - saves2['logits'].float()).abs().max()}"
        )

    def test_multiple_saves_isolation(self, model):
        """Two traces saving multiple variables — all should be isolated."""
        with model.trace(ET_PROMPT, serve=SERVE_URL, blocking=False) as t1:
            h5 = model.model.layers[5].output[0].save()
            h10 = model.model.layers[10].output[0].save()

        with model.trace(MSG_PROMPT, serve=SERVE_URL, blocking=False) as t2:
            h5 = model.model.layers[5].output[0].save()
            h10 = model.model.layers[10].output[0].save()

        saves1 = t1.collect(timeout=30)
        saves2 = t2.collect(timeout=30)

        # Each should have both keys
        assert "h5" in saves1 and "h10" in saves1
        assert "h5" in saves2 and "h10" in saves2

        # Different prompts → different activations
        assert not torch.equal(saves1["h5"], saves2["h5"])
        assert not torch.equal(saves1["h10"], saves2["h10"])


# =========================================================================
# 3. Multi-process clients
# =========================================================================

def _worker_fn(prompt, serve_url, model_name, result_queue):
    """Run a single serve trace in a separate process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    try:
        from nnsight.modeling.vllm import VLLM
        m = VLLM(model_name)
        with m.trace(prompt, temperature=0.0, top_p=1, serve=serve_url):
            logits = m.logits.output.save()
        token = m.tokenizer.decode(logits.argmax(dim=-1))
        result_queue.put(("ok", token))
    except Exception as e:
        result_queue.put(("error", str(e)))


class TestMultiProcessClients:
    """Multiple processes hitting the server concurrently."""

    def test_two_processes(self, model):
        """Two separate processes sending requests simultaneously."""
        ctx = mp.get_context("spawn")
        q1 = ctx.Queue()
        q2 = ctx.Queue()

        p1 = ctx.Process(target=_worker_fn, args=(ET_PROMPT, SERVE_URL, MODEL, q1))
        p2 = ctx.Process(target=_worker_fn, args=(MSG_PROMPT, SERVE_URL, MODEL, q2))

        t0 = time.perf_counter()
        p1.start()
        p2.start()

        p1.join(timeout=300)
        p2.join(timeout=300)
        elapsed = time.perf_counter() - t0

        if p1.is_alive() or p2.is_alive():
            p1.kill()
            p2.kill()
            pytest.skip("Multi-process test timed out (likely HuggingFace network issue in spawned processes)")

        status1, result1 = q1.get(timeout=5)
        status2, result2 = q2.get(timeout=5)

        assert status1 == "ok", f"Process 1 failed: {result1}"
        assert status2 == "ok", f"Process 2 failed: {result2}"

        assert "Paris" in result1, f"Process 1: expected 'Paris', got '{result1}'"
        assert "New" in result2, f"Process 2: expected 'New', got '{result2}'"

        print(f"\nTwo-process test completed in {elapsed:.1f}s")


# =========================================================================
# 4. Error handling for misuse
# =========================================================================

class TestErrorHandling:
    """Verify that misuse produces clear errors, not silent corruption."""

    def test_wrong_server_url(self, model):
        """serve= pointing to wrong port should raise ConnectionError."""
        with pytest.raises((ConnectionError, httpx.ConnectError)):
            with model.trace(ET_PROMPT, serve="http://127.0.0.1:9999"):
                model.logits.output.save()

    def test_nonblocking_collect_before_fire(self, model):
        """Calling collect() on a blocking trace should raise AttributeError."""
        with model.trace(ET_PROMPT, serve=SERVE_URL) as t:
            model.logits.output.save()

        with pytest.raises(AttributeError, match="no pending serve request"):
            t.collect()

    def test_nonblocking_double_collect(self, model):
        """Calling collect() twice should work (future.result() is idempotent)."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t:
            logits = model.logits.output.save()

        saves1 = t.collect(timeout=30)
        saves2 = t.collect(timeout=30)  # Should not raise

        assert saves1.keys() == saves2.keys()

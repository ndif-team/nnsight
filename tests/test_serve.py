"""Tests for nnsight-vllm-serve client (serve= parameter).

These tests MUST run against a pre-started nnsight-serve instance.
They verify that requests go to the server, not local execution.

Start the server:
    CUDA_VISIBLE_DEVICES=1 conda run -n ndif-dev python -m nnsight.modeling.vllm.serve.cli \
        Qwen/Qwen2.5-0.5B-Instruct --port 6679 --gpu-memory-utilization 0.3

Run tests (on a DIFFERENT GPU):
    CUDA_VISIBLE_DEVICES=2 conda run -n ndif-dev python -m pytest tests/test_serve.py -v -x -s
"""

import os
import sys

import pytest
import torch

# Force a GPU that is NOT used by the server.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

try:
    from nnsight.modeling.vllm import VLLM
except Exception as e:
    pytest.skip(f"Skipping serve tests (vllm import failed): {e}", allow_module_level=True)


SERVE_URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6679")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_LAYERS = 24
HIDDEN_DIM = 896

ET_PROMPT = "The Eiffel Tower is located in the city of"
MSG_PROMPT = "Madison Square Garden is located in the city of"


# =============================================================================
# Server reachability check — skip ALL tests if server is down
# =============================================================================

def _server_is_reachable() -> bool:
    """Check /health endpoint before running any tests."""
    import httpx
    try:
        r = httpx.get(f"{SERVE_URL}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False

if not _server_is_reachable():
    pytest.skip(
        f"nnsight-serve not reachable at {SERVE_URL}. Start the server first.",
        allow_module_level=True,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def model():
    """Meta-only VLLM model (no local engine, no dispatch)."""
    m = VLLM(MODEL)
    assert not m.dispatched, "Model should NOT be dispatched for serve tests"
    return m


@pytest.fixture(scope="module")
def local_model():
    """Locally-dispatched VLLM model for numerical comparison."""
    return VLLM(MODEL, dispatch=True, gpu_memory_utilization=0.3)


# =============================================================================
# Helper: verify the model stayed meta after a serve trace
# =============================================================================

def _assert_not_dispatched(model):
    """After a serve trace, the client model must still be meta (not dispatched)."""
    assert not model.dispatched, (
        "Model was dispatched locally during serve trace! "
        "The serve= parameter is not being intercepted — requests are running locally."
    )


# =============================================================================
# 1. Basic Inference
# =============================================================================

class TestBasicInference:
    """Basic activation capture and logit access via serve=."""

    def test_hidden_state_capture(self, model):
        """Capture a single layer's output."""
        with model.trace(ET_PROMPT, serve=SERVE_URL):
            hidden = model.model.layers[5].output[0].save()

        _assert_not_dispatched(model)
        assert hidden.shape[-1] == HIDDEN_DIM
        assert hidden.dtype == torch.bfloat16
        assert not torch.all(hidden == 0)

    def test_logit_prediction(self, model):
        """Verify next-token prediction via logits."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL):
            logits = model.logits.output.save()

        _assert_not_dispatched(model)
        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" in next_token, f"Expected 'Paris', got '{next_token}'"

    def test_multi_layer_capture(self, model):
        """Capture outputs from multiple layers simultaneously."""
        with model.trace(ET_PROMPT, serve=SERVE_URL):
            h0 = model.model.layers[0].output[0].save()
            h12 = model.model.layers[12].output[0].save()
            h23 = model.model.layers[23].output[0].save()

        _assert_not_dispatched(model)
        assert h0.shape == h12.shape == h23.shape
        assert not torch.equal(h0, h12)
        assert not torch.equal(h12, h23)


# =============================================================================
# 2. Interventions
# =============================================================================

class TestInterventions:
    """Activation modification via serve=."""

    def test_zero_out_mlp(self, model):
        """Zero out an MLP output and verify it takes effect."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL):
            out = model.model.layers[-2].mlp.output.clone()
            out[:] = 0
            model.model.layers[-2].mlp.output = out
            hs = model.model.layers[-2].mlp.output.save()
            logits = model.logits.output.save()

        _assert_not_dispatched(model)
        assert torch.all(hs == 0)
        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" not in next_token

    def test_swap_intervention(self, model):
        """Swap an MLP output with zeros using torch.zeros_like."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL):
            model.model.layers[-2].mlp.output = torch.zeros_like(
                model.model.layers[-2].mlp.output
            )
            hs = model.model.layers[-2].mlp.output.save()

        _assert_not_dispatched(model)
        assert torch.all(hs == 0)


# =============================================================================
# 3. Batched Multi-Invoke
# =============================================================================

class TestBatching:
    """Multiple prompts in a single trace via invoke()."""

    def test_batched_clean_and_corrupted(self, model):
        """Run clean and corrupted versions of the same prompt in one trace."""
        with model.trace(temperature=0.0, top_p=1, serve=SERVE_URL) as tracer:
            with tracer.invoke(ET_PROMPT):
                clean_hs = model.model.layers[-2].mlp.output.save()
                clean_logits = model.logits.output.save()

            with tracer.invoke(ET_PROMPT):
                out = model.model.layers[-2].mlp.output[:].clone()
                out[:] = 0
                model.model.layers[-2].mlp.output = out
                corrupted_hs = model.model.layers[-2].mlp.output.save()
                corrupted_logits = model.logits.output.save()

        _assert_not_dispatched(model)
        assert not torch.all(clean_hs == 0)
        assert torch.all(corrupted_hs == 0)

        clean_token = model.tokenizer.decode(clean_logits.argmax(dim=-1))
        corrupted_token = model.tokenizer.decode(corrupted_logits.argmax(dim=-1))
        assert "Paris" in clean_token
        assert "Paris" not in corrupted_token

    def test_two_different_prompts(self, model):
        """Capture logits from two different prompts in one trace."""
        with model.trace(temperature=0.0, top_p=1, serve=SERVE_URL) as tracer:
            with tracer.invoke(ET_PROMPT):
                et_logits = model.logits.output.save()
            with tracer.invoke(MSG_PROMPT):
                msg_logits = model.logits.output.save()

        _assert_not_dispatched(model)
        et_token = model.tokenizer.decode(et_logits.argmax(dim=-1))
        msg_token = model.tokenizer.decode(msg_logits.argmax(dim=-1))
        assert "Paris" in et_token
        assert "New" in msg_token


# =============================================================================
# 4. Cross-Invoke Shared State
# =============================================================================

class TestCrossInvokeSharedState:
    """Shared Python objects across invokes in the same trace."""

    def test_shared_list_across_invokes(self, model):
        """Collect logits from multiple invokes into a shared list."""
        prompts = [ET_PROMPT, MSG_PROMPT]

        with model.trace(temperature=0.0, top_p=1, serve=SERVE_URL) as tracer:
            out_ids = [list() for _ in range(len(prompts))].save()
            for i, prompt in enumerate(prompts):
                with tracer.invoke(prompt):
                    out_ids[i].append(model.logits.output.argmax(dim=-1))

        _assert_not_dispatched(model)
        assert len(out_ids) == 2
        assert len(out_ids[0]) == 1
        assert len(out_ids[1]) == 1

        et_token = model.tokenizer.decode(out_ids[0][0])
        msg_token = model.tokenizer.decode(out_ids[1][0])
        assert "Paris" in et_token
        assert "New" in msg_token


# =============================================================================
# 5. Token ID Inputs
# =============================================================================

class TestTokenInputs:
    """Non-string inputs: token ID lists and HF tokenizer dicts."""

    def test_token_id_list(self, model):
        """Pass token IDs directly instead of a string prompt."""
        token_ids = model.tokenizer.encode(ET_PROMPT)

        with model.trace(token_ids, temperature=0.0, top_p=1, serve=SERVE_URL):
            logits = model.logits.output.save()

        _assert_not_dispatched(model)
        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" in next_token

    def test_hf_tokenizer_dict(self, model):
        """Pass a HuggingFace tokenizer output dict."""
        hf_output = model.tokenizer(ET_PROMPT, return_tensors="pt")

        with model.trace(dict(hf_output), temperature=0.0, top_p=1, serve=SERVE_URL):
            logits = model.logits.output.save()

        _assert_not_dispatched(model)
        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" in next_token

    def test_token_ids_in_invoker(self, model):
        """Pass token IDs inside an invoker."""
        token_ids = model.tokenizer.encode(ET_PROMPT)

        with model.trace(temperature=0.0, top_p=1, serve=SERVE_URL) as tracer:
            with tracer.invoke(token_ids):
                logits = model.logits.output.save()

        _assert_not_dispatched(model)
        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" in next_token


# =============================================================================
# 6. Non-Blocking (Async) Mode
# =============================================================================

class TestNonBlocking:
    """Non-blocking serve requests via blocking=False."""

    def test_single_nonblocking(self, model):
        """Single non-blocking request returns saves via .collect()."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t:
            logits = model.logits.output.save()

        _assert_not_dispatched(model)
        saves = t.collect()
        assert "logits" in saves, f"Expected 'logits' in saves, got keys: {list(saves.keys())}"
        next_token = model.tokenizer.decode(saves["logits"].argmax(dim=-1))
        assert "Paris" in next_token, f"Expected 'Paris', got '{next_token}'"

    def test_concurrent_nonblocking(self, model):
        """Two non-blocking requests should run concurrently in vLLM."""
        import time

        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t1:
            logits1 = model.logits.output.save()
        with model.trace(MSG_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL, blocking=False) as t2:
            logits2 = model.logits.output.save()

        _assert_not_dispatched(model)

        # Both should be in-flight now. Collect results.
        saves1 = t1.collect(timeout=30)
        saves2 = t2.collect(timeout=30)

        assert "logits1" in saves1 or "logits" in saves1, f"saves1 keys: {list(saves1.keys())}"
        assert "logits2" in saves2 or "logits" in saves2, f"saves2 keys: {list(saves2.keys())}"

    def test_nonblocking_hidden_states(self, model):
        """Non-blocking capture of hidden states."""
        with model.trace(ET_PROMPT, serve=SERVE_URL, blocking=False) as t:
            hidden = model.model.layers[5].output[0].save()

        _assert_not_dispatched(model)
        saves = t.collect()
        assert "hidden" in saves, f"Expected 'hidden' in saves, got keys: {list(saves.keys())}"
        assert saves["hidden"].shape[-1] == HIDDEN_DIM


# =============================================================================
# 7. Numerical Comparison: serve vs local
# =============================================================================

class TestNumericalMatch:
    """Verify serve= produces the same results as local execution."""

    def test_hidden_states_match(self, model, local_model):
        """Hidden states from serve and local should be bitwise identical."""
        with model.trace(ET_PROMPT, serve=SERVE_URL):
            serve_h = model.model.layers[5].output[0].save()

        with local_model.trace(ET_PROMPT):
            local_h = local_model.model.layers[5].output[0].save()

        _assert_not_dispatched(model)
        assert serve_h.shape == local_h.shape
        assert torch.equal(serve_h.cpu(), local_h.cpu()), (
            f"Max diff: {(serve_h.cpu().float() - local_h.cpu().float()).abs().max()}"
        )

    def test_logits_match(self, model, local_model):
        """Logits from serve and local should be bitwise identical."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL):
            serve_logits = model.logits.output.save()

        with local_model.trace(ET_PROMPT, temperature=0.0, top_p=1):
            local_logits = local_model.logits.output.save()

        _assert_not_dispatched(model)
        assert torch.equal(serve_logits.cpu(), local_logits.cpu()), (
            f"Max diff: {(serve_logits.cpu().float() - local_logits.cpu().float()).abs().max()}"
        )

    def test_intervention_results_match(self, model, local_model):
        """Intervention results from serve and local should match."""
        def run_trace(m, **kwargs):
            with m.trace(ET_PROMPT, temperature=0.0, top_p=1, **kwargs):
                m.model.layers[-2].mlp.output = torch.zeros_like(
                    m.model.layers[-2].mlp.output
                )
                logits = m.logits.output.save()
            return logits

        serve_logits = run_trace(model, serve=SERVE_URL)
        local_logits = run_trace(local_model)

        _assert_not_dispatched(model)
        assert torch.equal(serve_logits.cpu(), local_logits.cpu()), (
            f"Max diff: {(serve_logits.cpu().float() - local_logits.cpu().float()).abs().max()}"
        )

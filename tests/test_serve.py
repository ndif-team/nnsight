"""Tests for nnsight-vllm-serve client (serve= parameter).

Adapted from test_vllm.py. All tests run against a pre-started
nnsight-serve instance. The client uses only a meta model (no local GPU
dispatch). The server must be started separately before running:

    CUDA_VISIBLE_DEVICES=1 python -m nnsight.modeling.vllm.serve.cli \
        Qwen/Qwen2.5-0.5B-Instruct --port 6679 --gpu-memory-utilization 0.3

Run tests:
    CUDA_VISIBLE_DEVICES=2 python -m pytest tests/test_serve.py -v -x

Architecture mapping (GPT-2 → Qwen2.5):
    GPT-2: model.transformer.h[i].mlp.output
    Qwen:  model.model.layers[i].mlp.output
    Both:  model.logits.output, model.samples.output
"""

import os
import subprocess
import sys

import pytest
import torch

# Use a GPU for the meta model init (vLLM needs CUDA info for attention backend).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

try:
    from nnsight.modeling.vllm import VLLM
except Exception as e:
    pytest.skip(f"Skipping serve tests: {e}", allow_module_level=True)


SERVE_URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6679")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_LAYERS = 24
HIDDEN_DIM = 896

# Qwen2.5-0.5B tokenizes "The Eiffel Tower is located in the city of" → 11 tokens
ET_PROMPT = "The Eiffel Tower is located in the city of"
MSG_PROMPT = "Madison Square Garden is located in the city of"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def model():
    """Meta-only VLLM model (no local engine dispatch)."""
    return VLLM(MODEL)


@pytest.fixture(scope="module")
def local_model():
    """Locally-dispatched VLLM model for comparison tests."""
    return VLLM(MODEL, dispatch=True, gpu_memory_utilization=0.3)


# =============================================================================
# 1. Basic Inference
# =============================================================================


class TestBasicInference:
    """Basic activation capture and logit access via serve=."""

    def test_hidden_state_capture(self, model):
        """Capture a single layer's output."""
        with model.trace(ET_PROMPT, serve=SERVE_URL):
            hidden = model.model.layers[5].output[0].save()

        assert hidden.shape[1] == HIDDEN_DIM
        assert hidden.dtype == torch.bfloat16
        assert not torch.all(hidden == 0)

    def test_logit_prediction(self, model):
        """Verify next-token prediction via logits."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL):
            logits = model.logits.output.save()

        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" in next_token

    def test_multi_layer_capture(self, model):
        """Capture outputs from multiple layers simultaneously."""
        with model.trace(ET_PROMPT, serve=SERVE_URL):
            h0 = model.model.layers[0].output[0].save()
            h12 = model.model.layers[12].output[0].save()
            h23 = model.model.layers[23].output[0].save()

        assert h0.shape == h12.shape == h23.shape
        # Different layers should produce different activations.
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

        assert torch.all(hs == 0)
        # Zeroing a layer should change the prediction.
        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" not in next_token

    def test_swap_intervention(self, model):
        """Swap an MLP output with zeros using torch.zeros_like."""
        with model.trace(ET_PROMPT, temperature=0.0, top_p=1, serve=SERVE_URL):
            model.model.layers[-2].mlp.output = torch.zeros_like(
                model.model.layers[-2].mlp.output
            )
            hs = model.model.layers[-2].mlp.output.save()

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

        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" in next_token

    def test_hf_tokenizer_dict(self, model):
        """Pass a HuggingFace tokenizer output dict."""
        hf_output = model.tokenizer(ET_PROMPT, return_tensors="pt")

        with model.trace(dict(hf_output), temperature=0.0, top_p=1, serve=SERVE_URL):
            logits = model.logits.output.save()

        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" in next_token

    def test_token_ids_in_invoker(self, model):
        """Pass token IDs inside an invoker."""
        token_ids = model.tokenizer.encode(ET_PROMPT)

        with model.trace(temperature=0.0, top_p=1, serve=SERVE_URL) as tracer:
            with tracer.invoke(token_ids):
                logits = model.logits.output.save()

        next_token = model.tokenizer.decode(logits.argmax(dim=-1))
        assert "Paris" in next_token


# =============================================================================
# 6. Numerical Comparison: serve vs local
# =============================================================================


class TestNumericalMatch:
    """Verify serve= produces the same results as local execution."""

    def test_hidden_states_match(self, model, local_model):
        """Hidden states from serve and local should be bitwise identical."""
        # Serve path.
        with model.trace(ET_PROMPT, serve=SERVE_URL):
            serve_h = model.model.layers[5].output[0].save()

        # Local path.
        with local_model.trace(ET_PROMPT):
            local_h = local_model.model.layers[5].output[0].save()

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

        assert torch.equal(serve_logits.cpu(), local_logits.cpu()), (
            f"Max diff: {(serve_logits.cpu().float() - local_logits.cpu().float()).abs().max()}"
        )

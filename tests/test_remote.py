"""
Tests for remote execution with NDIF.

These tests require:
- A valid NDIF API key
- The model to be running on NDIF (check nnsight.ndif_status())

Run with: pytest tests/test_remote.py -v -s
"""

import pytest
import torch
import nnsight
from nnsight import LanguageModel, CONFIG

# Configure API key for testing
# In production, use CONFIG.set_default_api_key() once
CONFIG.API.APIKEY = "<key here>"

# Check if model is available before running tests
MODEL_ID = "openai-community/gpt2"


@pytest.fixture(scope="module")
def model():
    """Load model on meta device (no weights loaded locally)."""
    m = LanguageModel(MODEL_ID)
    print(f"\nModel device: {m.device}")  # Should be 'meta'
    return m


@pytest.fixture(scope="module")
def model_available():
    """Check if model is running on NDIF.

    NOTE: is_model_running() may return False even if model is running.
    The actual test will fail with RemoteException if model is truly unavailable.
    """
    # Just check that NDIF service is reachable
    try:
        nnsight.ndif_status()
        return True
    except Exception as e:
        pytest.skip(f"NDIF service unreachable: {e}")


class TestBasicRemote:
    """Basic remote execution tests."""

    def test_simple_trace(self, model, model_available):
        """Simple remote trace - get model output."""
        with model.trace("The Eiffel Tower is in the city of", remote=True):
            logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

        print(f"\nPredicted token ID: {logit}")
        print(f"Decoded: {model.tokenizer.decode(logit)}")

        assert isinstance(logit, torch.Tensor)
        assert logit.dim() == 0  # Scalar

    def test_save_hidden_states(self, model, model_available):
        """Save hidden states from a layer."""
        with model.trace("Hello world", remote=True):
            hidden = model.transformer.h[5].output[0].save()

        print(f"\nHidden states shape: {hidden.shape}")

        assert hidden.dim() == 3  # [batch, seq, hidden]

    def test_multiple_saves(self, model, model_available):
        """Save multiple values."""
        with model.trace("Testing multiple saves", remote=True):
            h0 = model.transformer.h[0].output[0].save()
            h5 = model.transformer.h[5].output[0].save()
            h11 = model.transformer.h[11].output[0].save()
            logits = model.lm_head.output.save()

        print(f"\nLayer 0 shape: {h0.shape}")
        print(f"Layer 5 shape: {h5.shape}")
        print(f"Layer 11 shape: {h11.shape}")
        print(f"Logits shape: {logits.shape}")

        assert h0.shape == h5.shape == h11.shape


class TestPrintStatements:
    """Test that print statements are sent back as LOG."""

    def test_print_in_trace(self, model, model_available):
        """Print statements should appear as LOG status."""
        print("\n--- Watch for LOG messages in the output below ---")

        with model.trace("Hello", remote=True):
            hidden = model.transformer.h[0].output[0]
            print(f"Hidden shape: {hidden.shape}")
            print(f"Hidden mean: {hidden.mean()}")
            output = model.lm_head.output.save()

        print(f"Output shape: {output.shape}")


class TestInterventions:
    """Test interventions (modifications) remotely."""

    def test_zero_out_layer(self, model, model_available):
        """Zero out a layer's output."""
        # Get clean output first
        with model.trace("The Eiffel Tower is in the city of", remote=True):
            clean_logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

        # Now zero out layer 5
        with model.trace("The Eiffel Tower is in the city of", remote=True):
            model.transformer.h[5].output[0][:] = 0
            modified_logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

        print(f"\nClean prediction: {model.tokenizer.decode(clean_logit)}")
        print(f"Modified prediction: {model.tokenizer.decode(modified_logit)}")

        # They should be different (zeroing a layer changes output)
        # But not guaranteed - depends on model

    def test_activation_patching(self, model, model_available):
        """Patch activations from one prompt to another using session.

        IMPORTANT: Prompts must have the same token length for patching!
        """
        # Both prompts have 7 tokens:
        # "The Eiffel Tower is in" -> [464, 412, 733, 417, 8765, 318, 287]
        # "Shaquille O Neal plays sport" -> 7 tokens
        with model.session(remote=True):
            # First trace: get embeddings from "Paris" prompt
            with model.trace("The Eiffel Tower is in"):
                paris_embed = model.transformer.wte.output

            # Second trace: patch into different prompt (same token length!)
            with model.trace("Shaquille O Neal plays sport"):
                model.transformer.wte.output = paris_embed
                patched_logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

            # Third trace: get clean output for comparison
            with model.trace("Shaquille O Neal plays sport"):
                clean_logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

        print(f"\nClean prediction: {model.tokenizer.decode(clean_logit)}")
        print(f"Patched prediction: {model.tokenizer.decode(patched_logit)}")


class TestSessions:
    """Test session functionality."""

    def test_basic_session(self, model, model_available):
        """Multiple traces in one session."""
        with model.session(remote=True):
            with model.trace("Hello"):
                h1 = model.transformer.h[0].output[0].save()

            with model.trace("World"):
                h2 = model.transformer.h[0].output[0].save()

        print(f"\nTrace 1 hidden shape: {h1.shape}")
        print(f"Trace 2 hidden shape: {h2.shape}")

    def test_session_variable_sharing(self, model, model_available):
        """Variables from earlier traces accessible in later ones."""
        with model.session(remote=True):
            with model.trace("Source prompt"):
                source_hidden = model.transformer.h[5].output[0]  # No .save()!

            with model.trace("Target prompt"):
                # Can reference source_hidden directly
                diff = (model.transformer.h[5].output[0] - source_hidden).mean().save()

        print(f"\nMean difference: {diff}")
        assert isinstance(diff, torch.Tensor)


class TestGeneration:
    """Test multi-token generation."""

    def test_basic_generate(self, model, model_available):
        """Basic generation with remote=True."""
        with model.generate(
            "The quick brown fox", max_new_tokens=5, remote=True
        ) as tracer:
            output = tracer.result.save()

        decoded = model.tokenizer.decode(output[0])
        print(f"\nGenerated: {decoded}")

        assert len(output[0]) > 4  # Original + new tokens

    def test_generate_with_iter(self, model, model_available):
        """Access each generation step."""
        with model.generate("Hello", max_new_tokens=3, remote=True) as tracer:
            logits_list = list().save()

            with tracer.iter[:]:
                logits_list.append(model.lm_head.output[0][-1].argmax(dim=-1))

            output = tracer.result.save()

        print(f"\nGenerated tokens: {logits_list}")
        print(f"Full output: {model.tokenizer.decode(output[0])}")

        assert len(logits_list) == 3


class TestNonBlocking:
    """Test non-blocking execution."""

    def test_non_blocking_request(self, model, model_available):
        """Submit request without waiting."""
        with model.trace("Hello world", remote=True, blocking=False) as tracer:
            output = model.lm_head.output.save()

        # At this point, request is submitted but we haven't waited
        backend = tracer.backend
        print(f"\nJob ID: {backend.job_id}")
        print(f"Initial status: {backend.job_status}")

        # Poll for result
        import time

        max_attempts = 30
        for i in range(max_attempts):
            result = backend()
            if result is not None:
                print(f"Got result after {i+1} polls")
                # Result is a dict with variable names as keys
                print(f"Result keys: {result.keys()}")
                print(f"Output shape: {result['output'].shape}")
                break
            print(f"Polling... status: {backend.job_status}")
            time.sleep(1)
        else:
            pytest.fail("Job did not complete in time")


class TestTensorBestPractices:
    """Test best practices for saving tensors."""

    def test_detach_cpu_save(self, model, model_available):
        """Best practice: detach and move to CPU before saving."""
        with model.trace("Hello", remote=True):
            # Best practice for minimal download size
            hidden = model.transformer.h[0].output[0].detach().cpu().save()

        print(f"\nHidden device: {hidden.device}")
        print(f"Hidden shape: {hidden.shape}")

        assert hidden.device == torch.device("cpu")


class TestErrorCases:
    """Test error handling."""

    def test_out_of_order_remote(self, model, model_available):
        """Out of order access should still raise error remotely."""
        with pytest.raises(Exception):  # Could be OutOfOrderError or RemoteException
            with model.trace("Hello", remote=True):
                # Access layer 5 before layer 0
                h5 = model.transformer.h[5].output[0].save()
                h0 = model.transformer.h[0].output[0].save()  # Error!


# =============================================================================
# Manual exploration - run these interactively
# =============================================================================


def explore_ndif_status():
    """Check what models are available."""
    print("\n" + "=" * 60)
    print("NDIF Status:")
    print("=" * 60)
    print(nnsight.ndif_status())
    print()
    print(f"GPT-2 available: {nnsight.is_model_running('openai-community/gpt2')}")


def explore_basic_remote():
    """Basic remote execution exploration."""
    model = LanguageModel("openai-community/gpt2")
    print(f"Model device: {model.device}")

    with model.trace("The Eiffel Tower is in the city of", remote=True):
        logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

    print(f"Prediction: {model.tokenizer.decode(logit)}")


def explore_print_logging():
    """See how print statements come back."""
    model = LanguageModel("openai-community/gpt2")

    with model.trace("Hello world", remote=True):
        hidden = model.transformer.h[0].output[0]
        print("=== This should appear as LOG ===")
        print(f"Shape: {hidden.shape}")
        print(f"Mean: {hidden.mean()}")
        print(f"Std: {hidden.std()}")
        output = model.lm_head.output.save()

    print(f"\nFinal output shape: {output.shape}")


def explore_session():
    """Explore session with variable sharing."""
    model = LanguageModel("openai-community/gpt2")

    with model.session(remote=True):
        with model.trace("Megan Rapinoe plays the sport of"):
            hs = model.transformer.h[5].output[0][:, -1, :]  # No save needed

        with model.trace("Shaquille O'Neal plays the sport of"):
            clean = model.lm_head.output[0][-1].argmax(dim=-1).save()

        with model.trace("Shaquille O'Neal plays the sport of"):
            model.transformer.h[5].output[0][:, -1, :] = hs  # Patch!
            patched = model.lm_head.output[0][-1].argmax(dim=-1).save()

    print(f"Clean: {model.tokenizer.decode(clean)}")
    print(f"Patched: {model.tokenizer.decode(patched)}")


if __name__ == "__main__":
    print("Run with: pytest tests/test_remote.py -v -s")
    print("Or explore interactively:")
    print("  from tests.test_remote import *")
    print("  explore_ndif_status()")
    print("  explore_basic_remote()")
    print("  explore_print_logging()")
    print("  explore_session()")

"""
Tests for VisionLanguageModel functionality with LLaVA-Interleave-Qwen-0.5B.

These tests cover VLM-specific features:
- Loading and basic tracing with text + images
- Generation with vision inputs
- Activation modification
- Text-only fallback (no images)
- Batching with multiple invokes
- Scan mode

Note: LLaVA's underlying Qwen2 decoder layers return the hidden state
tensor directly (not a tuple), so `layer.output` is a 3D tensor
`[batch, seq, hidden]` rather than a tuple `(tensor, ...)` as in GPT-2.
"""

import pytest
import torch
import nnsight


# =============================================================================
# Basic Tracing with Images
# =============================================================================


class TestVLMBasic:
    """Tests for basic VLM tracing with text and images."""

    @torch.no_grad()
    def test_load_and_trace(self, vlm: nnsight.VisionLanguageModel, dummy_image):
        """Test that we can load a VLM and trace with text + image."""
        with vlm.trace("<image>\nDescribe this image", images=[dummy_image]):
            hidden = vlm.model.language_model.layers[-1].output.save()

        assert hidden is not None
        assert isinstance(hidden, torch.Tensor)
        assert hidden.ndim == 3

    @torch.no_grad()
    def test_processor_exists(self, vlm: nnsight.VisionLanguageModel):
        """Test that processor and tokenizer are loaded."""
        assert vlm.processor is not None
        assert vlm.tokenizer is not None

    @torch.no_grad()
    def test_hidden_state_shapes(self, vlm: nnsight.VisionLanguageModel, dummy_image):
        """Test that hidden states have expected dimensions."""
        with vlm.trace("<image>\nWhat is in this image?", images=[dummy_image]):
            first_hidden = vlm.model.language_model.layers[0].output.save()
            last_hidden = vlm.model.language_model.layers[-1].output.save()

        assert first_hidden.ndim == 3
        assert last_hidden.ndim == 3
        # Batch size should be 1
        assert first_hidden.shape[0] == 1
        assert last_hidden.shape[0] == 1
        # Hidden dim should match
        assert first_hidden.shape[2] == last_hidden.shape[2]


# =============================================================================
# Generation
# =============================================================================


class TestVLMGeneration:
    """Tests for VLM generation with images."""

    @torch.no_grad()
    def test_basic_generation(self, vlm: nnsight.VisionLanguageModel, dummy_image):
        """Test generation with text + image produces output."""
        with vlm.generate(
            "<image>\nDescribe this image", images=[dummy_image], max_new_tokens=5
        ) as tracer:
            output = vlm.generator.output.save()

        decoded = vlm.tokenizer.decode(output[0], skip_special_tokens=True)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    @torch.no_grad()
    def test_generation_with_invoke(
        self, vlm: nnsight.VisionLanguageModel, dummy_image
    ):
        """Test generation using explicit invoke."""
        with vlm.generate(max_new_tokens=3) as tracer:
            with tracer.invoke("<image>\nWhat is this?", images=[dummy_image]):
                output = vlm.generator.output.save()

        decoded = vlm.tokenizer.decode(output[0], skip_special_tokens=True)
        assert isinstance(decoded, str)
        assert len(decoded) > 0


# =============================================================================
# Activation Modification
# =============================================================================


class TestVLMActivationModification:
    """Tests for modifying activations in VLM traces."""

    @torch.no_grad()
    def test_zero_hidden_states(self, vlm: nnsight.VisionLanguageModel, dummy_image):
        """Test zeroing out hidden states changes output."""
        with vlm.trace("<image>\nDescribe this image", images=[dummy_image]):
            pre = vlm.model.language_model.layers[-1].output.clone().save()
            vlm.model.language_model.layers[-1].output[:] = 0
            post = vlm.model.language_model.layers[-1].output.save()

        assert not (pre == 0).all().item()
        assert (post == 0).all().item()

    @torch.no_grad()
    def test_save_and_access_input(self, vlm: nnsight.VisionLanguageModel, dummy_image):
        """Test saving layer inputs."""
        with vlm.trace("<image>\nDescribe this image", images=[dummy_image]):
            layer_input = vlm.model.language_model.layers[0].input.save()
            layer_output = vlm.model.language_model.layers[0].output.save()

        assert layer_input is not None
        assert isinstance(layer_input, torch.Tensor)
        assert layer_output is not None
        assert isinstance(layer_output, torch.Tensor)


# =============================================================================
# Text-Only Fallback
# =============================================================================


class TestVLMTextOnly:
    """Tests for VLM with text-only input (no images)."""

    @torch.no_grad()
    def test_text_only_trace(self, vlm: nnsight.VisionLanguageModel):
        """Test tracing with text only, no images."""
        with vlm.trace("Hello world"):
            hidden = vlm.model.language_model.layers[-1].output.save()

        assert hidden is not None
        assert isinstance(hidden, torch.Tensor)
        assert hidden.ndim == 3

    @torch.no_grad()
    def test_text_only_generation(self, vlm: nnsight.VisionLanguageModel):
        """Test generation with text only."""
        with vlm.generate("The capital of France is", max_new_tokens=3) as tracer:
            output = vlm.generator.output.save()

        decoded = vlm.tokenizer.decode(output[0], skip_special_tokens=True)
        assert isinstance(decoded, str)
        assert len(decoded) > 0


# =============================================================================
# Batching
# =============================================================================


class TestVLMBatching:
    """Tests for batching with VLM invokers."""

    @torch.no_grad()
    def test_multiple_invokes_with_images(
        self, vlm: nnsight.VisionLanguageModel, dummy_image
    ):
        """Test multiple invokes each with images."""
        from PIL import Image

        img2 = Image.new("RGB", (64, 64), color=(32, 128, 64))

        with vlm.trace() as tracer:
            with tracer.invoke("<image>\nDescribe image one", images=[dummy_image]):
                out_1 = vlm.lm_head.output[:, -1].save()

            with tracer.invoke("<image>\nDescribe image two", images=[img2]):
                out_2 = vlm.lm_head.output[:, -1].save()

        assert out_1.shape[0] == 1
        assert out_2.shape[0] == 1

    @torch.no_grad()
    def test_empty_invoke(self, vlm: nnsight.VisionLanguageModel, dummy_image):
        """Test empty invoke (promptless) after image invoke."""
        with vlm.trace() as tracer:
            with tracer.invoke("<image>\nDescribe this image", images=[dummy_image]):
                out_1 = vlm.lm_head.output[:, -1].save()

            with tracer.invoke():
                out_all = vlm.lm_head.output[:, -1].save()

        assert out_1.shape[0] == 1
        assert out_all.shape[0] == 1
        assert torch.equal(out_1, out_all)


# =============================================================================
# Scan Mode
# =============================================================================


@pytest.mark.scan
class TestVLMScan:
    """Tests for scan mode with VLM."""

    @torch.no_grad()
    def test_scan_with_image(self, vlm: nnsight.VisionLanguageModel, dummy_image):
        """Test scan mode with text + image."""
        with vlm.scan("<image>\nDescribe this image", images=[dummy_image]):
            out_shape = nnsight.save(vlm.model.language_model.layers[0].output.shape)

        assert len(out_shape) == 3
        assert out_shape[0] == 1

    @torch.no_grad()
    def test_scan_text_only(self, vlm: nnsight.VisionLanguageModel):
        """Test scan mode with text only."""
        with vlm.scan("Hello"):
            out_shape = nnsight.save(vlm.model.language_model.layers[0].output.shape)

        assert len(out_shape) == 3
        assert out_shape[0] == 1

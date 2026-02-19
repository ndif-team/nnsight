"""
Tests for DiffusionModel functionality.

These tests cover diffusion model specific features:
- Basic tracing (1-step default)
- Full pipeline generation
- Saving denoiser activations
- Interventions on denoiser activations
- Batching with multiple invokes
- Iteration over generation steps
- Seed reproducibility
- Non-tracing forward pass
- Meta loading (dispatch=False)
- Flux (transformer-based) pipeline support (optional, --test-flux)
"""

import pytest

pytest.importorskip("diffusers")

import torch
import PIL
import numpy as np

from nnsight import DiffusionModel


# =============================================================================
# Basic Tests
# =============================================================================


class TestDiffusionBasic:
    """Tests for basic DiffusionModel loading and single-step tracing."""

    @torch.no_grad()
    def test_trace_default_one_step(self, tiny_sd, sd_prompt):
        """Trace with default 1-step produces output with .images."""
        with tiny_sd.trace(sd_prompt) as tracer:
            output = tracer.result.save()

        assert hasattr(output, "images")
        assert len(output.images) >= 1

    @torch.no_grad()
    def test_trace_output_is_pil(self, tiny_sd, sd_prompt):
        """Trace output images are PIL Images."""
        with tiny_sd.trace(sd_prompt) as tracer:
            output = tracer.result.save()

        for img in output.images:
            assert isinstance(img, PIL.Image.Image)


# =============================================================================
# Generation Tests
# =============================================================================


class TestDiffusionGeneration:
    """Tests for .generate() with multiple inference steps."""

    @torch.no_grad()
    def test_generate_two_steps(self, tiny_sd, sd_prompt):
        """Generate with num_inference_steps=2 produces output images."""
        with tiny_sd.generate(sd_prompt, num_inference_steps=2) as tracer:
            output = tracer.result.save()

        assert hasattr(output, "images")
        assert len(output.images) >= 1
        for img in output.images:
            assert isinstance(img, PIL.Image.Image)


# =============================================================================
# Tracing Tests
# =============================================================================


class TestDiffusionTracing:
    """Tests for saving denoiser activations during tracing."""

    @torch.no_grad()
    def test_save_denoiser_output(self, tiny_sd, sd_prompt):
        """Can save the denoiser (UNet) output during a trace."""
        with tiny_sd.trace(sd_prompt) as tracer:
            denoiser_out = tiny_sd.unet.output.save()

        if isinstance(denoiser_out, tuple):
            assert isinstance(denoiser_out[0], torch.Tensor)
        else:
            assert isinstance(denoiser_out, torch.Tensor)


# =============================================================================
# Intervention Tests
# =============================================================================


class TestDiffusionIntervention:
    """Tests for intervening on denoiser activations."""

    @torch.no_grad()
    def test_zero_denoiser_changes_output(self, tiny_sd, sd_prompt):
        """Zeroing denoiser output changes the final image vs unmodified."""
        # Unmodified run
        with tiny_sd.trace(sd_prompt, num_inference_steps=1) as tracer:
            output_clean = tracer.result.save()

        # Modified run: zero out denoiser output
        with tiny_sd.trace(sd_prompt, num_inference_steps=1) as tracer:
            tiny_sd.unet.output[0][:] = 0
            output_modified = tracer.result.save()

        clean_arr = np.array(output_clean.images[0])
        modified_arr = np.array(output_modified.images[0])

        assert not np.array_equal(clean_arr, modified_arr)


# =============================================================================
# Batching Tests
# =============================================================================


class TestDiffusionBatching:
    """Tests for batching multiple prompts via invokes."""

    @torch.no_grad()
    def test_multiple_invokes(self, tiny_sd):
        """Multiple invokes with different prompts produce batched output."""
        with tiny_sd.trace() as tracer:
            with tracer.invoke("A cat"):
                out1 = tiny_sd.unet.output.save()

            with tracer.invoke("A dog"):
                out2 = tiny_sd.unet.output.save()

        if isinstance(out1, tuple):
            assert out1[0].shape[0] >= 1
            assert out2[0].shape[0] >= 1
        else:
            assert out1.shape[0] >= 1
            assert out2.shape[0] >= 1


# =============================================================================
# Iteration Tests
# =============================================================================


class TestDiffusionIteration:
    """Tests for iterating over generation steps."""

    @torch.no_grad()
    def test_iterate_steps(self, tiny_sd, sd_prompt):
        """Iterate over generation steps with tracer.iter[:] and collect denoiser outputs."""
        num_steps = 2
        with tiny_sd.generate(sd_prompt, num_inference_steps=num_steps) as tracer:
            denoiser_outputs = list().save()
            for step in tracer.iter[:]:
                denoiser_outputs.append(tiny_sd.unet.output[0].clone())

        assert len(denoiser_outputs) == num_steps


# =============================================================================
# Seed Tests
# =============================================================================


class TestDiffusionSeed:
    """Tests for seed reproducibility."""

    @torch.no_grad()
    def test_seed_reproducibility(self, tiny_sd, sd_prompt):
        """Same seed produces identical outputs."""
        with tiny_sd.generate(
            sd_prompt, num_inference_steps=2, seed=42
        ) as tracer:
            output1 = tracer.result.save()

        with tiny_sd.generate(
            sd_prompt, num_inference_steps=2, seed=42
        ) as tracer:
            output2 = tracer.result.save()

        arr1 = np.array(output1.images[0])
        arr2 = np.array(output2.images[0])

        assert np.array_equal(arr1, arr2)


# =============================================================================
# Text-Only (Non-Tracing Forward) Tests
# =============================================================================


class TestDiffusionTextOnly:
    """Tests for running the pipeline without a tracing context."""

    @torch.no_grad()
    def test_generate_no_context(self, tiny_sd, sd_prompt):
        """Generate without a with-context produces valid images."""
        output = tiny_sd.generate(
            sd_prompt, num_inference_steps=2
        )

        assert hasattr(output, "images")
        assert len(output.images) >= 1
        for img in output.images:
            assert isinstance(img, PIL.Image.Image)


# =============================================================================
# Meta Loading Tests (dispatch=False)
# =============================================================================


class TestDiffusionMetaLoading:
    """Tests for dispatch=False meta loading path."""

    def test_meta_loading_creates_meta_params(self):
        """dispatch=False creates model with meta-device parameters (no weights downloaded)."""
        model = DiffusionModel(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            safety_checker=None,
            dispatch=False,
        )

        # Model should exist and have a UNet
        assert model._model is not None
        assert hasattr(model._model, "unet")

        # Parameters should be on the meta device (no real weights)
        for param in model._model.unet.parameters():
            assert param.device.type == "meta"

    @torch.no_grad()
    def test_meta_loading_auto_dispatches_on_trace(self, sd_prompt):
        """dispatch=False model auto-dispatches real weights on first trace."""
        model = DiffusionModel(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            safety_checker=None,
            dispatch=False,
        )

        # Verify meta device before trace
        for param in model._model.unet.parameters():
            assert param.device.type == "meta"

        # Trace should trigger auto-dispatch and produce valid output
        with model.trace(sd_prompt) as tracer:
            output = tracer.result.save()

        assert hasattr(output, "images")
        assert len(output.images) >= 1

        # After trace, parameters should no longer be on meta device
        for param in model._model.unet.parameters():
            assert param.device.type != "meta"


# =============================================================================
# Flux (Transformer-Based) Tests — require --test-flux
# =============================================================================


class TestFluxBasic:
    """Tests for Flux (transformer-based) diffusion pipeline."""

    @torch.no_grad()
    def test_flux_trace(self, flux, sd_prompt):
        """Flux trace with 1-step default produces output with .images."""
        with flux.trace(sd_prompt) as tracer:
            output = tracer.result.save()

        assert hasattr(output, "images")
        assert len(output.images) >= 1
        for img in output.images:
            assert isinstance(img, PIL.Image.Image)

    @torch.no_grad()
    def test_flux_save_transformer_output(self, flux, sd_prompt):
        """Can save the transformer denoiser output during a Flux trace."""
        with flux.trace(sd_prompt) as tracer:
            denoiser_out = flux.transformer.output.save()

        if isinstance(denoiser_out, tuple):
            assert isinstance(denoiser_out[0], torch.Tensor)
        else:
            assert isinstance(denoiser_out, torch.Tensor)

    @torch.no_grad()
    def test_flux_generate(self, flux, sd_prompt):
        """Flux .generate() with explicit steps produces output images."""
        with flux.generate(sd_prompt, num_inference_steps=2) as tracer:
            output = tracer.result.save()

        assert hasattr(output, "images")
        assert len(output.images) >= 1

    @torch.no_grad()
    def test_flux_intervention(self, flux, sd_prompt):
        """Zeroing Flux transformer output changes the final image."""
        with flux.trace(sd_prompt, num_inference_steps=1) as tracer:
            output_clean = tracer.result.save()

        with flux.trace(sd_prompt, num_inference_steps=1) as tracer:
            flux.transformer.output[0][:] = 0
            output_modified = tracer.result.save()

        clean_arr = np.array(output_clean.images[0])
        modified_arr = np.array(output_modified.images[0])

        assert not np.array_equal(clean_arr, modified_arr)

    @torch.no_grad()
    def test_flux_iteration(self, flux, sd_prompt):
        """Iterate over Flux generation steps and collect transformer outputs."""
        num_steps = 2
        with flux.generate(sd_prompt, num_inference_steps=num_steps) as tracer:
            denoiser_outputs = list().save()
            for step in tracer.iter[:]:
                denoiser_outputs.append(flux.transformer.output[0].clone())

        assert len(denoiser_outputs) == num_steps

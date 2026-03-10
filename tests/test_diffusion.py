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


    @torch.no_grad()
    def test_text_encoder_batching(
        self,
        tiny_sd, 
        cat_prompt,
        panda_prompt,
        birthday_cake_prompt,
        wave_prompt
    ):
        with tiny_sd.generate(
            num_inference_steps=20,
            num_images_per_prompt=3,
            seed=423
        ) as tracer:

            with tracer.invoke(cat_prompt):
                encoder_out_1 = tiny_sd.text_encoder.text_model.encoder.layers[-1].output.save()

            with tracer.invoke([panda_prompt, birthday_cake_prompt]):
                encoder_out_2 = tiny_sd.text_encoder.text_model.encoder.layers[-1].output.save()

            with tracer.invoke(wave_prompt):
                encoder_out_3 = tiny_sd.text_encoder.text_model.encoder.layers[-1].output.save()

            with tracer.invoke():
                encoder_out_all = tiny_sd.text_encoder.text_model.encoder.layers[-1].output.save()

        assert encoder_out_all.shape[0] == encoder_out_1.shape[0] + encoder_out_2.shape[0] + encoder_out_3.shape[0]
        assert encoder_out_1.shape[0] == 1
        assert encoder_out_2.shape[0] == 2
        assert encoder_out_3.shape[0] == 1

        assert torch.all(encoder_out_1 == encoder_out_all[0:1])
        assert torch.all(encoder_out_2 == encoder_out_all[1:3])
        assert torch.all(encoder_out_3 == encoder_out_all[3:])

    @torch.no_grad()
    def test_unet_batching(
        self,
        tiny_sd, 
        cat_prompt,
        panda_prompt,
        birthday_cake_prompt,
        wave_prompt
    ):
        with tiny_sd.generate(
            num_inference_steps=20,
            num_images_per_prompt=3,
            guidance_scale=1.0,
            seed=423
        ) as tracer:

            with tracer.invoke(cat_prompt):
                unet_out_1 = tiny_sd.unet.output[0].save()

            with tracer.invoke([panda_prompt, birthday_cake_prompt]):
                unet_out_2 = tiny_sd.unet.output[0].save()

            with tracer.invoke(wave_prompt):
                unet_out_3 = tiny_sd.unet.output[0].save()

            with tracer.invoke():
                unet_out_all = tiny_sd.unet.output[0].save()

        assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
        assert unet_out_1.shape[0] == 3
        assert unet_out_2.shape[0] == 6
        assert unet_out_3.shape[0] == 3

        assert torch.all(unet_out_1 == unet_out_all[0:3])
        assert torch.all(unet_out_2 == unet_out_all[3:9])
        assert torch.all(unet_out_3 == unet_out_all[9:])

    @torch.no_grad()
    def test_unet_batching_with_guidance(
        self,
        tiny_sd, 
        cat_prompt, 
        panda_prompt, 
        birthday_cake_prompt, 
        wave_prompt
    ):
        with tiny_sd.generate(
            num_inference_steps=20,
            num_images_per_prompt=3,
            guidance_scale=7.5,
            seed=423
        ) as tracer:

            with tracer.invoke(cat_prompt):
                unet_out_1 = tiny_sd.unet.output[0].save()

            with tracer.invoke([panda_prompt, birthday_cake_prompt]):
                unet_out_2 = tiny_sd.unet.output[0].save()

            with tracer.invoke(wave_prompt):
                unet_out_3 = tiny_sd.unet.output[0].save()

            with tracer.invoke():
                unet_out_all = tiny_sd.unet.output[0].save()

        assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
        assert unet_out_1.shape[0] == 6
        assert unet_out_2.shape[0] == 12
        assert unet_out_3.shape[0] == 6

        assert torch.all(unet_out_1 == torch.cat([unet_out_all[0:3], unet_out_all[12:15]], dim=0))
        assert torch.all(unet_out_2 == torch.cat([unet_out_all[3:9], unet_out_all[15:21]], dim=0))
        assert torch.all(unet_out_3 == torch.cat([unet_out_all[9:12], unet_out_all[21:24]], dim=0))

    @torch.no_grad()
    def test_vae_batching(
        self,
        tiny_sd, 
        cat_prompt,
        panda_prompt,
        birthday_cake_prompt,
        wave_prompt
    ):
        with tiny_sd.generate(
            num_inference_steps=20,
            num_images_per_prompt=3,
            guidance_scale=1.0,
            seed=423
        ) as tracer:

            with tracer.invoke(cat_prompt):
                vae_decoder_out_1 = tiny_sd.vae.decoder.output.save()

            with tracer.invoke([panda_prompt, birthday_cake_prompt]):
                vae_decoder_out_2 = tiny_sd.vae.decoder.output.save()

            with tracer.invoke(wave_prompt):
                vae_decoder_out_3 = tiny_sd.vae.decoder.output.save()

            with tracer.invoke():
                vae_decoder_out_all = tiny_sd.vae.decoder.output.save()

        assert vae_decoder_out_all.shape[0] == vae_decoder_out_1.shape[0] + vae_decoder_out_2.shape[0] + vae_decoder_out_3.shape[0]
        assert vae_decoder_out_1.shape[0] == 3
        assert vae_decoder_out_2.shape[0] == 6
        assert vae_decoder_out_3.shape[0] == 3

        assert torch.all(vae_decoder_out_1 == vae_decoder_out_all[0:3])
        assert torch.all(vae_decoder_out_2 == vae_decoder_out_all[3:9])
        assert torch.all(vae_decoder_out_3 == vae_decoder_out_all[9:])

    @torch.no_grad()
    def test_text_encoder_swapping(
        self,
        tiny_sd, 
        cat_prompt,
        panda_prompt,
        birthday_cake_prompt,
        wave_prompt
    ):
        module = tiny_sd.text_encoder.text_model.encoder.layers[-1]

        with tiny_sd.generate(
            num_inference_steps=20,
            num_images_per_prompt=3,
            seed=423
        ) as tracer:

            with tracer.invoke(cat_prompt):
                module.output = torch.ones_like(module.output) * 0
                encoder_out_1 = module.output.save()

            with tracer.invoke([panda_prompt, birthday_cake_prompt]):
                module.output = torch.ones_like(module.output) * 1
                encoder_out_2 = module.output.save()

            with tracer.invoke(wave_prompt):
                module.output = torch.ones_like(module.output) * 2
                encoder_out_3 = module.output.save()
            
            with tracer.invoke():
                encoder_out_all = module.output.save()

        assert encoder_out_all.shape[0] == encoder_out_1.shape[0] + encoder_out_2.shape[0] + encoder_out_3.shape[0]
        assert encoder_out_1.shape[0] == 1
        assert encoder_out_2.shape[0] == 2
        assert encoder_out_3.shape[0] == 1

        assert torch.all(encoder_out_1 == encoder_out_all[0:1])
        assert torch.all(encoder_out_2 == encoder_out_all[1:3])
        assert torch.all(encoder_out_3 == encoder_out_all[3:])
        

    @torch.no_grad()
    def test_unet_swapping(
        self,
        tiny_sd, 
        cat_prompt,
        panda_prompt,
        birthday_cake_prompt,
        wave_prompt
    ):
        with tiny_sd.generate(
            num_inference_steps=20,
            num_images_per_prompt=3,
            guidance_scale=1.0,
            seed=423
        ) as tracer:

            with tracer.invoke(cat_prompt):
                tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 0,)
                unet_out_1 = tiny_sd.unet.output[0].save()

            with tracer.invoke([panda_prompt, birthday_cake_prompt]):
                tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 1,)
                unet_out_2 = tiny_sd.unet.output[0].save()

            with tracer.invoke(wave_prompt):
                tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 2,)
                unet_out_3 = tiny_sd.unet.output[0].save()

            with tracer.invoke():
                unet_out_all = tiny_sd.unet.output[0].save()

        assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
        assert unet_out_1.shape[0] == 3
        assert unet_out_2.shape[0] == 6
        assert unet_out_3.shape[0] == 3

        assert torch.all(unet_out_1 == unet_out_all[0:3])
        assert torch.all(unet_out_2 == unet_out_all[3:9])
        assert torch.all(unet_out_3 == unet_out_all[9:])


    @torch.no_grad()
    def test_unet_swapping_with_guidance(
        self,
        tiny_sd, 
        cat_prompt,
        panda_prompt,
        birthday_cake_prompt,
        wave_prompt
    ):
        with tiny_sd.generate(
            num_inference_steps=20,
            num_images_per_prompt=3,
            guidance_scale=7.5,
            seed=423
        ) as tracer:

            with tracer.invoke(cat_prompt):
                tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 0,)
                unet_out_1 = tiny_sd.unet.output[0].save()

            with tracer.invoke([panda_prompt, birthday_cake_prompt]):
                tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 1,)
                unet_out_2 = tiny_sd.unet.output[0].save()

            with tracer.invoke(wave_prompt):
                tiny_sd.unet.output = (torch.ones_like(tiny_sd.unet.output[0]) * 2,)
                unet_out_3 = tiny_sd.unet.output[0].save()

            with tracer.invoke():
                unet_out_all = tiny_sd.unet.output[0].save()

        assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
        assert unet_out_1.shape[0] == 6
        assert unet_out_2.shape[0] == 12
        assert unet_out_3.shape[0] == 6

        assert torch.all(unet_out_1 == torch.cat([unet_out_all[0:3], unet_out_all[12:15]], dim=0))
        assert torch.all(unet_out_2 == torch.cat([unet_out_all[3:9], unet_out_all[15:21]], dim=0))
        assert torch.all(unet_out_3 == torch.cat([unet_out_all[9:12], unet_out_all[21:24]], dim=0))


    @torch.no_grad()
    def test_vae_swapping(
        self,
        tiny_sd,
        cat_prompt,
        panda_prompt,
        birthday_cake_prompt,
        wave_prompt
    ):
        with tiny_sd.generate(
            num_inference_steps=20,
            num_images_per_prompt=3,
            guidance_scale=1.0,
            seed=423
        ) as tracer:

            with tracer.invoke(cat_prompt):
                tiny_sd.vae.decoder.output = torch.ones_like(tiny_sd.vae.decoder.output) * 0
                unet_out_1 = tiny_sd.vae.decoder.output.save()

            with tracer.invoke([panda_prompt, birthday_cake_prompt]):
                tiny_sd.vae.decoder.output = torch.ones_like(tiny_sd.vae.decoder.output) * 1
                unet_out_2 = tiny_sd.vae.decoder.output.save()

            with tracer.invoke(wave_prompt):
                tiny_sd.vae.decoder.output = torch.ones_like(tiny_sd.vae.decoder.output) * 2
                unet_out_3 = tiny_sd.vae.decoder.output.save()

            with tracer.invoke():
                unet_out_all = tiny_sd.vae.decoder.output.save()

        assert unet_out_all.shape[0] == unet_out_1.shape[0] + unet_out_2.shape[0] + unet_out_3.shape[0]
        assert unet_out_1.shape[0] == 3
        assert unet_out_2.shape[0] == 6
        assert unet_out_3.shape[0] == 3

        assert torch.all(unet_out_1 == unet_out_all[0:3])
        assert torch.all(unet_out_2 == unet_out_all[3:9])
        assert torch.all(unet_out_3 == unet_out_all[9:])


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
            "segmind/tiny-sd",
            torch_dtype=torch.float16,
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
            "segmind/tiny-sd",
            torch_dtype=torch.float16,
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

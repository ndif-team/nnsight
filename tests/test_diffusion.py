"""Tests on a real ``DiffusionModel`` (a tiny stable-diffusion).

Covers the lazy meta build, the envoy tree over the pipeline's module components,
the two run modes (both run the whole pipeline; ``trace`` defaults to one denoising
step, ``generate`` to the pipeline's default), running a single component directly
(``model.unet.trace(...)``), reading and modifying activations, ``seed=``
reproducibility, and multi-prompt batching via ``DiffusionBatcher``
(classifier-free-guidance doubling and ``num_images_per_prompt``).
"""


import nnsight
import pytest
import torch

pytest.importorskip("diffusers")

import numpy

from nnsight.modeling.diffusion import DiffusionModel, _resolve_component_class

REPO = "hf-internal-testing/tiny-stable-diffusion-torch"
PROMPT = "a photo of a cat"
KWARGS = dict(num_inference_steps=2, output_type="np")

# Denoiser architectures beyond the UNet SD above: transformer-based (Flux, SD3)
# and a UNet with dual text encoders (SDXL), each a different pipeline/component set.
ARCHITECTURES = [
    ("hf-internal-testing/tiny-flux-pipe", "transformer"),
    ("hf-internal-testing/tiny-sd3-pipe", "transformer"),
    ("hf-internal-testing/tiny-sdxl-pipe", "unet"),
]


@pytest.fixture(scope="module")
def sd():
    return DiffusionModel(REPO)


@pytest.fixture(scope="module", params=ARCHITECTURES, ids=["flux", "sd3", "sdxl"])
def arch(request):
    repo, denoiser = request.param
    return DiffusionModel(repo), denoiser


def _grey():
    """A plain input image, for the image-to-image pipelines."""
    from PIL import Image

    return Image.new("RGB", (64, 64), (127, 127, 127))


def denoiser_inputs(model, batch=1):
    """Build one denoiser (unet) forward's inputs from its config."""
    unet = model.unet
    sample = torch.randn(
        batch, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size
    )
    timestep = torch.tensor(1.0)
    encoder_hidden_states = torch.randn(batch, 4, unet.config.cross_attention_dim)
    return sample, timestep, encoder_hidden_states


class TestBuild:
    def test_lazy_meta_build(self):
        model = DiffusionModel(REPO)
        assert model.dispatched is False
        # Module components are on meta and exposed as envoys.
        assert next(model.unet.parameters()).device.type == "meta"
        for name in ("unet", "vae", "text_encoder"):
            assert hasattr(model, name)
        # The tree nests down to real leaf modules.
        assert isinstance(model.unet.conv_in._module, torch.nn.Module)

    def test_dispatch_loads_real_weights(self):
        model = DiffusionModel(REPO)
        model.dispatch()
        assert model.dispatched is True
        assert next(model.unet.parameters()).device.type != "meta"
        assert model.pipeline is model._module.pipeline


class TestGeneration:
    @torch.no_grad()
    def test_generate_output_is_images(self, sd):
        with sd.generate(PROMPT, **KWARGS):
            out = sd.output.save()
        assert out.images.shape == (1, 128, 128, 3)

    @torch.no_grad()
    def test_generate_dispatches_lazily(self):
        model = DiffusionModel(REPO)
        with model.generate(PROMPT, **KWARGS):
            model.output.save()
        assert model.dispatched is True

    @torch.no_grad()
    def test_eager_generate_without_block(self, sd):
        out = sd.generate(PROMPT, **KWARGS)
        assert out.images.shape == (1, 128, 128, 3)


class TestTrace:
    @torch.no_grad()
    def test_trace_runs_pipeline(self, sd):
        # trace runs the whole pipeline; model.output is its image output.
        with sd.trace(PROMPT, output_type="np"):
            out = sd.output.save()
        assert out.images.shape == (1, 128, 128, 3)

    @torch.no_grad()
    def test_trace_defaults_to_one_step(self, sd):
        # No num_inference_steps -> a single denoising step (one unet call).
        with sd.trace(PROMPT, output_type="np") as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[:]:
                steps.append(sd.unet.output[0])
        assert len(steps) == 1


class TestComponentTrace:
    @torch.no_grad()
    def test_unet_trace_runs_denoiser(self, sd):
        # Run one forward of the denoiser on its own by tracing that envoy.
        sd.dispatch()  # a child-envoy trace doesn't dispatch the model itself
        sample, timestep, enc = denoiser_inputs(sd)
        with sd.unet.trace(sample, timestep, encoder_hidden_states=enc):
            out = sd.unet.output.save()
        assert out.sample.shape == sample.shape  # UNet2DConditionOutput


class TestInterventions:
    @torch.no_grad()
    def test_read_unet_activation(self, sd):
        with sd.generate(PROMPT, **KWARGS):
            # unet runs with return_dict=False, so its output is a tuple; under
            # classifier-free guidance it carries an uncond and a cond row.
            unet_out = sd.unet.output[0].save()
        assert isinstance(unet_out, torch.Tensor)
        assert unet_out.shape[0] == 2

    @torch.no_grad()
    def test_modify_unet_output_changes_image(self, sd):
        import numpy as np

        with sd.generate(PROMPT, **KWARGS):
            baseline = sd.output.save()
        with sd.generate(PROMPT, **KWARGS):
            sd.unet.output[0][:] = 0
            zeroed = sd.output.save()
        assert not np.allclose(baseline.images, zeroed.images)


class TestAutomodel:
    """`automodel=` chooses the diffusers class the weights are loaded through.

    `DiffusionPipeline.from_pretrained` builds whatever class the repo's
    `model_index.json` declares, which is the text-to-image one for every Stable
    Diffusion repo whatever the caller meant to do. The same weights serve other
    tasks through a different class, so naming one is how you reach them.
    """

    def test_the_default_is_the_declared_class(self):
        model = DiffusionModel(REPO, dispatch=True, safety_checker=None)
        assert type(model.pipeline).__name__ == "StableDiffusionPipeline"

    def test_a_concrete_class(self):
        from diffusers import StableDiffusionImg2ImgPipeline

        model = DiffusionModel(
            REPO, automodel=StableDiffusionImg2ImgPipeline, dispatch=True,
            safety_checker=None,
        )
        assert isinstance(model.pipeline, StableDiffusionImg2ImgPipeline)

    def test_a_class_named_as_a_string(self):
        model = DiffusionModel(
            REPO, automodel="StableDiffusionImg2ImgPipeline", dispatch=True,
            safety_checker=None,
        )
        assert type(model.pipeline).__name__ == "StableDiffusionImg2ImgPipeline"

    def test_an_auto_class_resolves_per_architecture(self):
        # `AutoPipelineForImage2Image` refuses to be constructed directly and
        # picks its class at load, so the meta build falls back to the declared
        # one. Both hold the same components, so the envoy tree stays valid.
        from diffusers import AutoPipelineForImage2Image

        model = DiffusionModel(REPO, automodel=AutoPipelineForImage2Image,
                               safety_checker=None)
        assert type(model.pipeline).__name__ == "StableDiffusionPipeline"

        with model.trace(PROMPT, image=_grey(), strength=0.5, **KWARGS):
            denoised = model.unet.output[0].save()

        assert type(model.pipeline).__name__ == "StableDiffusionImg2ImgPipeline"
        assert denoised.shape[0] == 2   # classifier-free guidance doubles the batch

    def test_an_intervention_reaches_an_image_to_image_run(self):
        from diffusers import AutoPipelineForImage2Image

        model = DiffusionModel(REPO, automodel=AutoPipelineForImage2Image,
                               dispatch=True, safety_checker=None)

        with model.trace(PROMPT, image=_grey(), strength=0.5, **KWARGS):
            base = model.output.save()
        with model.trace(PROMPT, image=_grey(), strength=0.5, **KWARGS):
            model.unet.output[0][:] = 0
            zeroed = model.output.save()

        assert not numpy.allclose(base.images[0], zeroed.images[0])


class TestSeed:
    @torch.no_grad()
    def test_seed_is_reproducible(self, sd):
        import numpy as np

        a = sd.generate(PROMPT, seed=7, **KWARGS)
        b = sd.generate(PROMPT, seed=7, **KWARGS)
        assert np.allclose(a.images, b.images)

    @torch.no_grad()
    def test_different_seeds_differ(self, sd):
        import numpy as np

        a = sd.generate(PROMPT, seed=7, **KWARGS)
        b = sd.generate(PROMPT, seed=8, **KWARGS)
        assert not np.allclose(a.images, b.images)

    @torch.no_grad()
    def test_seed_gives_each_image_its_own(self, sd):
        # A single seed fans out to one generator per image (seed + i), so the two
        # images differ but the run as a whole stays reproducible.
        import numpy as np

        out = sd.generate(PROMPT, seed=7, num_images_per_prompt=2, **KWARGS)
        assert out.images.shape[0] == 2
        assert not np.allclose(out.images[0], out.images[1])
        again = sd.generate(PROMPT, seed=7, num_images_per_prompt=2, **KWARGS)
        assert np.allclose(out.images, again.images)


class TestBatching:
    @torch.no_grad()
    def test_multi_prompt_narrows_unet(self, sd):
        # Two prompts -> the denoiser sees 2 x guidance = 4 rows; each invoke's
        # view is narrowed to its own uncond + cond (2 rows).
        with sd.generate(**KWARGS) as tracer:
            with tracer.invoke("a cat"):
                a = sd.unet.output[0].save()
            with tracer.invoke("a dog"):
                b = sd.unet.output[0].save()
        assert a.shape[0] == 2 and b.shape[0] == 2

    @torch.no_grad()
    def test_num_images_per_prompt_scales_rows(self, sd):
        # With 2 images per prompt, each invoke's denoiser view is 2 images x
        # guidance = 4 rows.
        with sd.generate(num_images_per_prompt=2, **KWARGS) as tracer:
            with tracer.invoke("a cat"):
                a = sd.unet.output[0].save()
            with tracer.invoke("a dog"):
                b = sd.unet.output[0].save()
        assert a.shape[0] == 4 and b.shape[0] == 4

    @torch.no_grad()
    def test_batched_edit_is_isolated(self, sd):
        # Zeroing one invoke's denoiser rows must not touch the other's.
        with sd.generate(**KWARGS) as tracer:
            with tracer.invoke("a cat"):
                sd.unet.output[0][:] = 0
            with tracer.invoke("a dog"):
                dog = sd.unet.output[0].save()
        assert bool((dog != 0).any())


# ---------------------------------------------------------------------------
# DiffusionModel behaviors these tests rely on:
#   * `trace` and `generate` both run the whole pipeline; `model.output` is its
#     image output. They differ only in the default num_inference_steps (trace 1).
#   * Run a single component's forward by tracing that envoy (`model.unet.trace`).
#   * `seed=` is turned into a reproducible `generator` (per-image for a batch).
#   * Reading a *batched* pipeline result object per-invoke isn't supported: a
#     required-field ModelOutput (StableDiffusionPipelineOutput) can't be rebuilt
#     by the row-narrowing walk. Batched interventions read component tensors.
#   * Op-level `.source` on the unet's forward is unavailable for this model (its
#     forward closes over free variables -> SourceNotAvailable), so there are no
#     diffusion-source tests here.
# ---------------------------------------------------------------------------


class TestIteration:
    @torch.no_grad()
    def test_iter_captures_each_denoiser_step(self, sd):
        with sd.generate(PROMPT, **KWARGS) as tracer:
            outs = nnsight.save([])
            for _ in tracer.iter[:]:
                outs.append(sd.unet.output[0])
        # The unet is invoked at least once per inference step.
        assert len(outs) >= KWARGS["num_inference_steps"]
        assert all(isinstance(o, torch.Tensor) for o in outs)


class TestSkip:
    @torch.no_grad()
    def test_skip_component_changes_image(self, sd):
        import numpy as np

        # Capture the conv_in output shape, then bypass conv_in with zeros.
        with sd.generate(PROMPT, num_inference_steps=1, output_type="np"):
            conv_in = sd.unet.conv_in.output.save()
        with sd.generate(PROMPT, **KWARGS):
            base = sd.output.save()
        with sd.generate(PROMPT, **KWARGS):
            sd.unet.conv_in.skip(torch.zeros_like(conv_in))
            skipped = sd.output.save()
        assert not np.allclose(base.images, skipped.images)


class TestCache:
    @torch.no_grad()
    def test_cache_unet(self, sd):
        with sd.generate(PROMPT, **KWARGS) as tracer:
            cache = tracer.cache(modules=[sd.unet]).save()
        keys = list(cache.keys())
        assert len(keys) >= 1
        assert cache[keys[0]].output is not None


class TestRename:
    @torch.no_grad()
    def test_rename_component(self):
        model = DiffusionModel(REPO, rename={"unet": "denoiser"})
        with model.generate(PROMPT, **KWARGS):
            denoised = model.denoiser.output[0].save()
        assert isinstance(denoised, torch.Tensor) and denoised.ndim == 4


class TestOutputTypes:
    @torch.no_grad()
    def test_default_output_is_pil(self, sd):
        import PIL

        out = sd.generate(PROMPT, num_inference_steps=2)  # default output_type
        assert isinstance(out.images[0], PIL.Image.Image)


class TestRemoteSimulation:
    """``remote="local"`` serializes the trace, deserializes it against local
    persistent objects, and runs it — the dry run of a real NDIF request."""

    def test_pipeline_is_a_persistent_object(self, sd):
        sd.dispatch()
        state = sd.__getstate__()
        assert state["pipeline"]._persistent_id == "Pipeline"
        assert sd._remoteable_persistent_objects()["Pipeline"] is sd.pipeline

    @torch.no_grad()
    def test_generate_round_trips(self, sd):
        with sd.generate(PROMPT, remote="local", **KWARGS):
            unet_out = sd.unet.output[0].save()
        assert isinstance(unet_out, torch.Tensor) and unet_out.shape[0] == 2


class TestComponentResolution:
    """The meta build resolves component classes across libraries and framework
    variants (regression for pipeline-subpackage and Flax/TF handling)."""

    def test_diffusers_and_transformers_libraries(self):
        import diffusers
        import transformers

        assert (
            _resolve_component_class("diffusers", "UNet2DConditionModel")
            is diffusers.UNet2DConditionModel
        )
        assert (
            _resolve_component_class("transformers", "CLIPTextModel")
            is transformers.CLIPTextModel
        )

    def test_pipeline_subpackage_library(self):
        # A component recorded as e.g. ["stable_diffusion", "..."] — the library is a
        # diffusers pipeline subpackage, not a top-level importable module.
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

        assert (
            _resolve_component_class("stable_diffusion", "StableDiffusionSafetyChecker")
            is StableDiffusionSafetyChecker
        )

    def test_flax_and_tf_names_resolve_to_pytorch(self):
        import diffusers
        import transformers

        assert (
            _resolve_component_class("diffusers", "FlaxUNet2DConditionModel")
            is diffusers.UNet2DConditionModel
        )
        assert (
            _resolve_component_class("transformers", "TFCLIPTextModel")
            is transformers.CLIPTextModel
        )

    def test_unresolvable_returns_none(self):
        assert _resolve_component_class(None, None) is None
        assert _resolve_component_class("diffusers", "NoSuchClass") is None
        assert _resolve_component_class("no_such_subpackage", "Whatever") is None


class TestArchitectures:
    """DiffusionModel across denoiser architectures: transformer (Flux, SD3) and
    UNet (SDXL), each a different pipeline with a different component set."""

    def test_lazy_build_exposes_denoiser(self, arch):
        model, denoiser = arch
        # The meta build produced a tree with this pipeline's denoiser component.
        assert hasattr(model, denoiser)

    @torch.no_grad()
    def test_generate_produces_image(self, arch):
        model, denoiser = arch
        with model.generate("a cat", num_inference_steps=1, output_type="np"):
            out = model.output.save()
        assert out.images.ndim == 4

    @torch.no_grad()
    def test_denoiser_intervention_changes_image(self, arch):
        import numpy as np

        model, denoiser = arch
        with model.generate(
            "a cat", num_inference_steps=1, output_type="np",
            generator=torch.Generator().manual_seed(0),
        ):
            base = model.output.save()
        with model.generate(
            "a cat", num_inference_steps=1, output_type="np",
            generator=torch.Generator().manual_seed(0),
        ):
            getattr(model, denoiser).output[0][:] = 0
            edited = model.output.save()
        assert not np.allclose(base.images, edited.images)

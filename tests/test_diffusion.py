"""Tests on a real ``DiffusionModel`` (a tiny stable-diffusion).

Covers the lazy meta build, the envoy tree over the pipeline's module components,
the two run modes (both run the whole pipeline; ``trace`` defaults to one denoising
step, ``generate`` to the pipeline's default), running a single component directly
(``model.unet.trace(...)``), reading and modifying activations, ``seed=``
reproducibility, multi-prompt batching via ``DiffusionBatcher``
(classifier-free-guidance doubling and ``num_images_per_prompt``), and
``automodel=`` — the diffusers class the weights load through, which is what
reaches a task other than the one the repo declares.
"""


from types import SimpleNamespace

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


def _mask():
    """A mask for the inpainting pipelines: a white square marks what to repaint."""
    from PIL import Image

    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (16, 16, 48, 48))
    return mask


# Each `AutoPipelineFor*` that reaches a task on this repo's architecture
# (stable-diffusion), the concrete class it picks, and the extra inputs that task
# takes — built per call, since the pipelines consume the image. The fourth Auto
# class, `AutoPipelineForText2Audio`, has no entry for this architecture and is
# covered by `TestText2Audio`.
AUTO_TASKS = [
    ("AutoPipelineForText2Image", "StableDiffusionPipeline", dict),
    (
        "AutoPipelineForImage2Image",
        "StableDiffusionImg2ImgPipeline",
        lambda: dict(image=_grey(), strength=0.6),
    ),
    (
        "AutoPipelineForInpainting",
        "StableDiffusionInpaintPipeline",
        lambda: dict(image=_grey(), mask_image=_mask()),
    ),
]
AUTO_IDS = ["text2image", "image2image", "inpainting"]


@pytest.fixture(scope="module", params=AUTO_TASKS, ids=AUTO_IDS)
def auto(request):
    """A lazily-built model loaded through one `AutoPipelineFor*`.

    What the meta build produced is captured here, before any test dispatches it,
    so the assertions about it don't depend on which test runs first.
    """
    import diffusers

    name, concrete, inputs = request.param
    model = DiffusionModel(REPO, automodel=getattr(diffusers, name), safety_checker=None)
    return SimpleNamespace(
        model=model,
        name=name,
        concrete=concrete,
        inputs=inputs,
        meta_class=type(model.pipeline).__name__,
        meta_components=set(model.pipeline.components),
    )


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

    def test_automodel_stays_out_of_the_load_kwargs(self):
        # It's a keyword-only argument of __init__, so it never joins the kwargs
        # replayed into `from_pretrained` on dispatch — where diffusers would
        # reject it as an unexpected component name.
        from diffusers import AutoPipelineForImage2Image

        model = DiffusionModel(REPO, automodel=AutoPipelineForImage2Image,
                               safety_checker=None)
        assert "automodel" not in model.kwargs
        assert model.automodel is AutoPipelineForImage2Image

    # `DiffusionPipeline` is a subclass of itself, so a guard that only asks
    # "was an automodel requested, and is it a pipeline?" lets the abstract base
    # through, and building from it filters every component out. Naming the
    # default explicitly is what a caller writes when threading a class through a
    # variable, so it has to mean the same as not naming one.
    def test_naming_the_default_class_explicitly_is_the_default(self):
        # `automodel=DiffusionPipeline` says exactly what `automodel=None` means,
        # and is what a caller passing a class through a variable ends up with.
        from diffusers import DiffusionPipeline

        model = DiffusionModel(REPO, automodel=DiffusionPipeline, safety_checker=None)
        assert type(model.pipeline).__name__ == "StableDiffusionPipeline"

    def test_the_default_class_explicitly_still_works_on_dispatch(self):
        # The same argument on the eager path: `_load` calls `from_pretrained` on
        # it, which is what the default does anyway, so this one is unaffected.
        from diffusers import DiffusionPipeline

        model = DiffusionModel(REPO, automodel=DiffusionPipeline, dispatch=True,
                               safety_checker=None)
        assert type(model.pipeline).__name__ == "StableDiffusionPipeline"


class TestAutoPipelines:
    """The `AutoPipelineFor*` classes, which pick a concrete class per architecture.

    Three of the four reach a real task on this repo's weights (text-to-image,
    image-to-image, inpainting); `AutoPipelineForText2Audio` has no entry for this
    architecture and is covered by `TestText2Audio`. None of them can be
    constructed directly, so the meta build falls back to the class the repo
    declares — which is only safe while that class holds the same components, so
    that is pinned here too.
    """

    @pytest.mark.parametrize("name", [task[0] for task in AUTO_TASKS], ids=AUTO_IDS)
    def test_the_meta_build_falls_back_to_the_declared_class(self, name):
        import diffusers

        model = DiffusionModel(REPO, automodel=getattr(diffusers, name),
                               safety_checker=None)
        assert type(model.pipeline).__name__ == "StableDiffusionPipeline"
        # The envoy tree is built from the components, so it has to be whole.
        for component in ("unet", "vae", "text_encoder"):
            assert hasattr(model, component)

    @torch.no_grad()
    def test_dispatch_loads_the_task_class(self, auto):
        with auto.model.trace(PROMPT, **auto.inputs(), **KWARGS):
            denoised = auto.model.unet.output[0].save()
        assert auto.meta_class == "StableDiffusionPipeline"
        assert type(auto.model.pipeline).__name__ == auto.concrete
        assert denoised.shape[0] == 2  # classifier-free guidance doubles the batch

    @torch.no_grad()
    def test_the_task_class_holds_the_same_components(self, auto):
        # The envoy tree is built once, off the meta pipeline, and re-pointed at
        # the dispatched one. A task class that took a different set of components
        # would leave envoys pointing at meta modules the real run never calls.
        auto.model.dispatch()
        assert set(auto.model.pipeline.components) == auto.meta_components

    @torch.no_grad()
    def test_an_intervention_changes_the_task_output(self, auto):
        with auto.model.trace(PROMPT, **auto.inputs(), **KWARGS):
            base = auto.model.output.save()
        with auto.model.trace(PROMPT, **auto.inputs(), **KWARGS):
            auto.model.unet.output[0][:] = 0
            zeroed = auto.model.output.save()
        assert not numpy.allclose(base.images[0], zeroed.images[0])

    @torch.no_grad()
    @pytest.mark.parametrize("repo,denoiser", ARCHITECTURES, ids=["flux", "sd3", "sdxl"])
    def test_image_to_image_picks_the_class_for_each_architecture(self, repo, denoiser):
        # The point of an Auto class over a concrete one: the same line reaches the
        # image-to-image task on Flux, SD3 and SDXL, whose task classes differ.
        from diffusers import AutoPipelineForImage2Image

        model = DiffusionModel(repo, automodel=AutoPipelineForImage2Image)
        components = set(model.pipeline.components)
        with model.trace(PROMPT, image=_grey(), strength=0.6, **KWARGS):
            denoised = getattr(model, denoiser).output[0].save()
        assert type(model.pipeline).__name__.endswith("Img2ImgPipeline")
        assert set(model.pipeline.components) == components
        assert isinstance(denoised, torch.Tensor)


class TestText2Audio:
    """`AutoPipelineForText2Audio`, the fourth Auto class and the odd one.

    There is no end-to-end test. The only tiny text-to-audio checkpoint on the Hub
    (`dn6/dummy-audioldm2`, 8MB) does not load in this environment — its T5
    tokenizer is a sentencepiece model that transformers 5 routes to a tiktoken
    reader that rejects it — and AudioLDM2 itself calls
    `GPT2Model._update_model_kwargs_for_generation`, removed in transformers 5, so
    the pipeline does not run under plain diffusers either. Loaded by hand from a
    patched copy it does reach nnsight intact: the tree carries the pipeline's
    `unet` (an `AudioLDM2UNet2DConditionModel`), which traces and takes an
    intervention. Its lazy path is unreachable for a separate reason — CLAP's
    encoder calls `.item()` in its constructor, which a meta tensor refuses — so an
    audio pipeline needs `dispatch=True` whatever `automodel=` says.

    What holds without a checkpoint is below.
    """

    def test_it_has_no_entry_for_an_image_architecture(self):
        # The refusal can only arrive at dispatch: the lazy build never consults
        # the mapping, it falls back to the class the repo declares.
        from diffusers import AutoPipelineForText2Audio

        model = DiffusionModel(REPO, automodel=AutoPipelineForText2Audio,
                               safety_checker=None)
        assert type(model.pipeline).__name__ == "StableDiffusionPipeline"
        with pytest.raises(ValueError, match="can't find a pipeline"):
            model.dispatch()

    def test_it_can_only_ever_pick_the_class_the_repo_declares(self):
        # Where the image mappings hold a different class per task, every
        # text-to-audio architecture has exactly one pipeline. So this Auto class
        # resolves to the declared class and never changes the task — which makes
        # the meta build's fallback to that class exactly right rather than merely
        # component-compatible.
        from diffusers.pipelines.auto_pipeline import (
            AUTO_TEXT2AUDIO_PIPELINES_MAPPING,
            _get_task_class,
        )

        for pipeline_cls in AUTO_TEXT2AUDIO_PIPELINES_MAPPING.values():
            assert (
                _get_task_class(AUTO_TEXT2AUDIO_PIPELINES_MAPPING, pipeline_cls.__name__)
                is pipeline_cls
            )


class TestAutomodelErrors:
    """What a name or class that can't load looks like, and when it is raised.

    The lazy build sees only what it can construct, so anything the *loader* has to
    judge — an Auto class with no entry, a class that isn't a pipeline at all —
    surfaces at dispatch rather than at construction.
    """

    def test_a_name_that_is_not_in_diffusers(self):
        with pytest.raises(AttributeError, match="NoSuchPipeline"):
            DiffusionModel(REPO, automodel="NoSuchPipeline")

    def test_a_class_needing_components_the_repo_has_not_got(self):
        # Asked for a concrete class, the meta build assembles *that* class from
        # the repo's declared components, so a mismatch is caught while building.
        with pytest.raises(TypeError, match="text_encoder_2"):
            DiffusionModel(REPO, automodel="StableDiffusionXLPipeline")

    def test_a_class_that_is_not_a_pipeline_survives_the_meta_build(self):
        # The meta guard only asks whether the class can assemble a pipeline, and
        # ignores it when it can't — so a non-pipeline is not refused, it is
        # deferred to `from_pretrained`, where the message is about a missing
        # config file rather than about `automodel=`.
        model = DiffusionModel(REPO, automodel="UNet2DConditionModel")
        assert type(model.pipeline).__name__ == "StableDiffusionPipeline"
        with pytest.raises(OSError):
            model.dispatch()


class TestAutomodelInteractions:
    """`automodel=` against the rest of DiffusionModel: renaming, single-component
    traces, batched invokes and the remote round trip."""

    @torch.no_grad()
    def test_rename_reaches_the_task_pipeline_s_denoiser(self):
        from diffusers import AutoPipelineForImage2Image

        model = DiffusionModel(REPO, automodel=AutoPipelineForImage2Image,
                               rename={"unet": "denoiser"}, safety_checker=None)
        with model.trace(PROMPT, image=_grey(), strength=0.6, **KWARGS):
            denoised = model.denoiser.output[0].save()
        assert type(model.pipeline).__name__ == "StableDiffusionImg2ImgPipeline"
        assert isinstance(denoised, torch.Tensor) and denoised.ndim == 4

    @torch.no_grad()
    def test_a_single_component_trace_after_an_auto_load(self):
        # The denoiser is the same module whichever task class holds it, so
        # tracing that envoy on its own is unaffected by the class swap.
        from diffusers import AutoPipelineForImage2Image

        model = DiffusionModel(REPO, automodel=AutoPipelineForImage2Image,
                               dispatch=True, safety_checker=None)
        sample, timestep, enc = denoiser_inputs(model)
        with model.unet.trace(sample, timestep, encoder_hidden_states=enc):
            out = model.unet.output.save()
        assert out.sample.shape == sample.shape

    @torch.no_grad()
    def test_batched_invokes_narrow_the_denoiser(self):
        # DiffusionBatcher's guidance-doubled layout is the denoiser's, not the
        # pipeline's, so it holds for an image-to-image run too — with one edit
        # confined to its own invoke's rows.
        from diffusers import AutoPipelineForImage2Image

        model = DiffusionModel(REPO, automodel=AutoPipelineForImage2Image,
                               dispatch=True, safety_checker=None)
        with model.trace(image=_grey(), strength=0.6, **KWARGS) as tracer:
            with tracer.invoke("a cat"):
                model.unet.output[0][:] = 0
            with tracer.invoke("a dog"):
                dog = model.unet.output[0].save()
        assert dog.shape[0] == 2
        assert bool((dog != 0).any())

    @torch.no_grad()
    def test_the_trace_round_trips_through_serialization(self):
        # `automodel` lands in __getstate__'s state, so it has to be picklable —
        # a diffusers class is, being module-level.
        from diffusers import AutoPipelineForImage2Image

        model = DiffusionModel(REPO, automodel=AutoPipelineForImage2Image,
                               dispatch=True, safety_checker=None)
        with model.trace(PROMPT, image=_grey(), strength=0.6, remote="local", **KWARGS):
            denoised = model.unet.output[0].save()
        assert isinstance(denoised, torch.Tensor) and denoised.shape[0] == 2


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

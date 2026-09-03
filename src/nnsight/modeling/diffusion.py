from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any, Optional

import torch

from ..intervention.batching import Batcher, BatchGroup
from ..intervention.envoy import traceable
from .huggingface import HuggingFaceModel
from .mixins.meta import MetaDevice

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline


def _resolve_component_class(library: Optional[str], class_name: Optional[str]) -> Optional[type]:
    """Resolve a pipeline component's class from its ``[library, class_name]`` spec.

    ``model_index.json`` records each component this way. The library is usually
    ``"diffusers"`` or ``"transformers"``, but can also be a diffusers pipeline
    subpackage (e.g. ``"stable_diffusion"`` -> ``diffusers.pipelines.stable_diffusion``
    for a safety checker). A Flax/TF class name resolves to its PyTorch equivalent,
    since the meta build is always PyTorch. Returns ``None`` when the spec names no
    class (a disabled component) or the class can't be found.
    """
    import diffusers
    import transformers

    if class_name is None:
        return None
    # Flax/TF variants share the PyTorch class's name without the prefix.
    for prefix in ("Flax", "TF"):
        if class_name.startswith(prefix):
            class_name = class_name[len(prefix):]
            break
    if library == "diffusers":
        return getattr(diffusers, class_name, None)
    if library == "transformers":
        return getattr(transformers, class_name, None)
    try:
        module = importlib.import_module(f"diffusers.pipelines.{library}")
    except (ImportError, RuntimeError):
        return None
    return getattr(module, class_name, None)


class _PipelineModule(torch.nn.Module):
    """An ``nn.Module`` view over a ``DiffusionPipeline`` whose forward runs it.

    A diffusion pipeline is not itself a module — it orchestrates several
    (unet/transformer, vae, text_encoder, ...). This registers each module
    component as a child so the whole tree is reachable as envoys
    (``model.unet``, ...), and forwards a call to the pipeline's denoising loop, so
    ``model.output`` is the pipeline's result. A ``seed`` is turned into a
    ``generator`` here (a per-image list for a batch), where the final batch size
    is known.
    """

    def __init__(self, pipeline: "DiffusionPipeline") -> None:
        super().__init__()
        # Not a Module, so this is a plain attribute — never a registered child.
        self.pipeline = pipeline
        for name, component in pipeline.components.items():
            if isinstance(component, torch.nn.Module):
                self.add_module(name, component)

    def forward(self, *args: Any, seed: Optional[int] = None, **kwargs: Any) -> Any:
        if seed is not None and kwargs.get("generator") is None:
            kwargs["generator"] = self._generator(seed, args, kwargs)
        return self.pipeline(*args, **kwargs)

    def _generator(self, seed: int, args: tuple, kwargs: dict) -> Any:
        # One generator per image; a batch of prompts gets `seed + i` per image so
        # each is independently reproducible (mirrors the batched-prompt offset).
        prompt = kwargs.get("prompt", args[0] if args else None)
        embeds = kwargs.get("prompt_embeds")
        if embeds is not None:
            batch = embeds.shape[0]
        elif isinstance(prompt, (list, tuple)):
            batch = len(prompt)
        else:
            batch = 1
        total = batch * kwargs.get("num_images_per_prompt", 1)
        device = self.pipeline.device
        if total > 1:
            return [torch.Generator(device).manual_seed(seed + i) for i in range(total)]
        return torch.Generator(device).manual_seed(seed)


class DiffusionBatcher(Batcher):
    """Batcher for the denoiser's guidance-doubled, multi-image batch layout.

    Where the base batcher assumes every batched activation is a plain dim-0 stack
    of the invokes' rows, a diffusion denoiser sees an expanded batch: each prompt
    is repeated ``num_images_per_prompt`` times, and under classifier-free guidance
    the whole thing is doubled (an unconditional half followed by a conditional
    half). This maps each invoke's plain ``[start, size]`` group onto that layout,
    picking the case by the tensor's leading dim at run time — so an intervention on
    ``model.unet`` reads (and writes) exactly its invoke's rows across both halves.
    """

    def __init__(self, envoy: Any, kwargs: Optional[dict] = None) -> None:
        super().__init__(envoy, kwargs)
        self.num_images: int = self.kwargs.get("num_images_per_prompt", 1)
        # Plain group start -> [start, size] in the image-expanded (pre-guidance)
        # batch. Filled as invokes are added; the sum of sizes is `image_total`.
        self.image_groups: dict[int, list] = {}
        self.image_total = 0

    def add(self, *inputs: Any, **kwargs: Any) -> BatchGroup:
        group = super().add(*inputs, **kwargs)
        if group is not None:
            _, size = group
            image_size = size * self.num_images
            self.image_groups[group[0]] = [self.image_total, image_size]
            self.image_total += image_size
        return group

    def _narrow_tensor(self, tensor: torch.Tensor, group: list) -> torch.Tensor:
        rows = tensor.shape[0]
        if rows == self.total:  # a plain (un-expanded) activation
            start, size = group
            return tensor.narrow(0, start, size)
        if rows == self.image_total:  # repeated num_images_per_prompt times
            start, size = self.image_groups[group[0]]
            return tensor.narrow(0, start, size)
        if rows == self.image_total * 2:  # + classifier-free guidance (uncond|cond)
            start, size = self.image_groups[group[0]]
            uncond = tensor.narrow(0, start, size)
            cond = tensor.narrow(0, start + self.image_total, size)
            return torch.cat([uncond, cond], dim=0)
        return tensor

    def _widen_tensor(
        self, full: torch.Tensor, group: list, edited: torch.Tensor
    ) -> torch.Tensor:
        rows = full.shape[0]
        if rows == self.total:
            start, size = group
            return self._splice(full, start, size, edited)
        if rows == self.image_total:
            start, size = self.image_groups[group[0]]
            return self._splice(full, start, size, edited)
        if rows == self.image_total * 2:  # edited holds both halves, uncond then cond
            start, size = self.image_groups[group[0]]
            uncond, cond = edited.chunk(2, dim=0)
            full = self._splice(full, start, size, uncond)
            return self._splice(full, start + self.image_total, size, cond)
        return full

    @staticmethod
    def _splice(
        full: torch.Tensor, start: int, size: int, edited: torch.Tensor
    ) -> torch.Tensor:
        pre = full.narrow(0, 0, start)
        post = full.narrow(0, start + size, full.shape[0] - start - size)
        return torch.cat([pre, edited, post], dim=0)


class DiffusionModel(HuggingFaceModel):
    """A model backed by a ``diffusers.DiffusionPipeline``.

    A diffusion pipeline orchestrates several modules (unet/transformer, vae,
    text_encoder, ...) around a denoising loop. This wraps the whole pipeline as one
    nnsight model: the pipeline is exposed as `pipeline`, each of its module
    components as an envoy (``model.unet``, ``model.vae``, ...), and interventions
    apply to any component along the way.

    Both `trace` and `generate` run the whole pipeline, with
    interventions firing on every component the denoising loop invokes;
    ``model.output`` (and ``tracer.result``) is the pipeline's output object.

    A traced run defaults to ``num_inference_steps=1`` — a fast one-step pass for
    inspecting or editing activations — and ``with model.generate(...):`` is a
    traced run, so it takes that default too; only a bare ``model.generate(...)``
    call outside a trace uses the pipeline's own default. The same call means one
    denoising step inside a ``with`` and fifty outside it, so pass
    ``num_inference_steps=`` whenever the image itself matters. To run one
    component's forward on its own, trace that envoy directly —
    ``with model.unet.trace(sample, timestep, encoder_hidden_states=...):``.

    On dispatch a real pipeline is built with real weights. The lazy meta build
    can't load a pipeline without weights, so each module component is constructed
    from its config on the meta device while the light components (scheduler,
    tokenizer, ...) load normally, and a meta pipeline of the same shape is
    assembled from them.

    Requires the optional ``diffusers`` package.

    Args:
        repo_id: A ``diffusers`` pipeline repo id (or local path) to load.
        *args: Forwarded to the mixin chain.
        **kwargs: Forwarded to the mixin chain and, on dispatch, to
            ``DiffusionPipeline.from_pretrained``. ``rename=`` exposes components
            under different envoy names (e.g. ``{"unet": "denoiser"}``);
            ``dispatch=True`` loads real weights immediately instead of building
            lazily on the meta device.

    Examples:
        >>> from nnsight.modeling.diffusion import DiffusionModel
        >>> model = DiffusionModel("hf-internal-testing/tiny-stable-diffusion-torch")
        >>> with model.generate("a photo of a cat", num_inference_steps=2) as tracer:
        ...     latents = model.unet.output[0].save()   # per denoising step
        ...     images = model.output.save()
        >>> images.images[0]  # a PIL image
    """

    pipeline: Optional["DiffusionPipeline"]

    def __init__(self, repo_id: Any, *args: Any, **kwargs: Any) -> None:
        self.pipeline = None
        super().__init__(repo_id, *args, **kwargs)

    # -- loading -------------------------------------------------------------

    def _meta_pipeline(self, repo_id: str) -> "DiffusionPipeline":
        # Build the pipeline component-by-component so no weights are loaded:
        # module components come from their config on meta (see MetaDevice),
        # everything else (schedulers, tokenizers, processors) loads normally.
        import diffusers

        config = diffusers.DiffusionPipeline.load_config(
            repo_id, revision=self.revision
        )
        pipeline_cls = getattr(diffusers, config["_class_name"])
        expected = set(inspect.signature(pipeline_cls.__init__).parameters) - {"self"}

        components: dict[str, Any] = {}
        for name, spec in config.items():
            # Each component is recorded as [library_name, class_name]; scalars
            # like `_class_name` or `requires_safety_checker` aren't components.
            if name.startswith("_") or not isinstance(spec, list) or name not in expected:
                continue
            library, class_name = spec
            klass = _resolve_component_class(library, class_name)
            if klass is None:
                # Unresolvable (or null) component — leave it out, as diffusers
                # would for a disabled one (e.g. a safety checker).
                components[name] = None
                continue
            components[name] = self._build_component(klass, repo_id, name)

        return pipeline_cls(**components)

    def _build_component(self, klass: type, repo_id: str, name: str) -> Any:
        import transformers

        # Module components carry the weights, so build them empty from config.
        if isinstance(klass, type) and issubclass(klass, torch.nn.Module):
            if hasattr(klass, "from_config") and hasattr(klass, "load_config"):
                # diffusers ModelMixin / ConfigMixin.
                sub_config = klass.load_config(
                    repo_id, subfolder=name, revision=self.revision
                )
                return klass.from_config(sub_config)
            # transformers PreTrainedModel.
            sub_config = transformers.AutoConfig.from_pretrained(
                repo_id, subfolder=name, revision=self.revision
            )
            return klass._from_config(sub_config)

        # Everything else is light (schedulers, tokenizers, processors) — load it
        # fully, on a real device: their constructors may move buffers around,
        # which a meta tensor can't do.
        with MetaDevice.real():
            return klass.from_pretrained(repo_id, subfolder=name, revision=self.revision)

    def _load_meta(self, repo_id: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
        self.pipeline = self._meta_pipeline(repo_id)
        return _PipelineModule(self.pipeline)

    def _load(self, repo_id: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
        from diffusers import DiffusionPipeline

        self.pipeline = DiffusionPipeline.from_pretrained(
            repo_id, revision=self.revision, **kwargs
        )
        return _PipelineModule(self.pipeline)

    # -- running -------------------------------------------------------------

    def trace(self, *inputs: Any, **kwargs: Any):
        """Trace the whole pipeline, defaulting to a single denoising step.

        ``num_inference_steps=1`` unless overridden — a fast one-step pass for
        inspecting or editing activations. ``model.output`` is the pipeline's
        output object. ``with model.generate(...):`` routes here as well, so a
        traced generation takes the one-step default too.

        Examples:
            >>> with model.trace("a photo of a cat"):
            ...     latents = model.unet.output[0].save()
            ...     images = model.output.save()
        """
        kwargs.setdefault("num_inference_steps", 1)
        return super().trace(*inputs, **kwargs)

    @traceable
    def generate(self, *inputs: Any, **kwargs: Any) -> Any:
        """Run the diffusion pipeline, returning its output object.

        ``with model.generate(...):`` traces the whole pipeline, so the block's
        interventions run against every component the denoising loop invokes (use
        ``tracer.iter`` to target a particular inference step); calling it directly
        just runs the pipeline. A traced run goes through `trace`, so it
        defaults to one denoising step where a direct call takes the pipeline's own
        default — name ``num_inference_steps`` to fix the count either way. The
        return value is the pipeline's own output
        object — read the images off its ``.images`` (or off ``model.output`` /
        ``tracer.result`` inside a trace).

        Examples:
            >>> with model.generate("a photo of a cat", num_inference_steps=20):
            ...     images = model.output.save()
            >>> images.images[0]  # a PIL image

        Args:
            *inputs: The pipeline's inputs, e.g. a text prompt.
            **kwargs: Forwarded to the pipeline, e.g. ``num_inference_steps``,
                ``guidance_scale``, ``num_images_per_prompt``, ``output_type``.
                ``seed`` (int) is turned into a reproducible ``generator`` — a
                per-image list for a batch; pass ``generator`` directly to override.

        Returns:
            The pipeline's output object; its ``.images`` holds the generated images.
        """
        # Dispatch is handled by interleave when tracing; the pipeline (and the
        # `seed` -> `generator` conversion) runs through the root _PipelineModule.
        return self._module(*inputs, **kwargs)

    # -- batching ------------------------------------------------------------

    _batcher_class = DiffusionBatcher

    def _batch_size(self, *inputs: Any, **kwargs: Any) -> int:
        """Number of batch rows an invoke's input contributes.

        A prompt is one row per string (a list of prompts is one per entry);
        ``prompt_embeds`` is one per leading row. Zero rows means params only (e.g.
        ``num_inference_steps=``), so the trace expects ``invoke()`` blocks.
        """
        value = inputs[0] if inputs else kwargs.get("prompt", kwargs.get("prompt_embeds"))
        if value is None:
            return 0
        if isinstance(value, str):
            return 1
        if isinstance(value, torch.Tensor):
            return 1 if value.ndim == 0 else value.shape[0]
        if isinstance(value, (list, tuple)):
            return len(value)
        return 1

    def _batch(self, invokes: list, fn: Any) -> tuple:
        """Merge invokes' prompts (or ``prompt_embeds``) into one pipeline call."""
        prompts: list = []
        embeds: list = []
        forward: dict = {}
        for inputs, kwargs in invokes:
            data = inputs[0] if inputs else kwargs.get("prompt")
            embed = kwargs.get("prompt_embeds")
            if embed is not None:
                embeds.append(embed if embed.ndim == 3 else embed.unsqueeze(0))
            elif isinstance(data, str):
                prompts.append(data)
            elif data is not None:
                prompts.extend(data)
            forward.update(
                {k: v for k, v in kwargs.items() if k not in ("prompt", "prompt_embeds")}
            )
        if embeds:
            forward["prompt_embeds"] = torch.cat(embeds, dim=0)
        else:
            forward["prompt"] = prompts
        return tuple(), forward

    # -- remote --------------------------------------------------------------

    def _remoteable_persistent_objects(self) -> dict:
        objects = super()._remoteable_persistent_objects()
        # The pipeline (and its light components) resolve to the server's live one;
        # its module components are already tagged by Envoy.__getstate__.
        if self.pipeline is not None:
            objects["Pipeline"] = self.pipeline
        return objects

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        # Reference the pipeline by persistent id instead of serializing it; the
        # server resolves it to its live object (see _remoteable_persistent_objects).
        if self.pipeline is not None:
            self.pipeline._persistent_id = "Pipeline"
        return state

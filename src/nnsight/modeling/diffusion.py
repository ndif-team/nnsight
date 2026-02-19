from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

import torch
from diffusers import DiffusionPipeline, pipelines
from transformers import PreTrainedTokenizerBase

from .. import util
from .huggingface import HuggingFaceModel


def _resolve_component_cls(lib_name: str, cls_name: str):
    """Resolve a pipeline component class from its library and class name.

    The ``model_index.json`` config stores each component as
    ``[library_name, class_name]``.  Library names can be ``"diffusers"``,
    ``"transformers"``, or a diffusers pipeline subpackage name like
    ``"stable_diffusion"``.

    If the class name starts with ``"Flax"`` or ``"TF"`` (JAX/TensorFlow
    variants), this function strips the prefix and resolves the
    corresponding PyTorch class instead.

    Returns:
        The resolved class, or ``None`` if it cannot be found.
    """
    import diffusers as _diffusers
    import transformers as _transformers

    # Normalize Flax/TF class names to their PyTorch equivalents
    for prefix in ("Flax", "TF"):
        if cls_name.startswith(prefix):
            cls_name = cls_name[len(prefix):]
            break

    if lib_name == "diffusers":
        return getattr(_diffusers, cls_name, None)
    elif lib_name == "transformers":
        return getattr(_transformers, cls_name, None)
    else:
        try:
            mod = importlib.import_module(f"diffusers.pipelines.{lib_name}")
            return getattr(mod, cls_name, None)
        except (ImportError, RuntimeError):
            return None


def _build_pipeline_from_config(
    automodel: type,
    repo_id: str,
    revision: Optional[str] = None,
    **kwargs,
) -> DiffusionPipeline:
    """Build a diffusion pipeline with meta-device ``nn.Module`` components.

    Downloads only the pipeline's ``model_index.json`` config and each
    component's ``config.json`` (a few KB total).  Each ``nn.Module``
    component is instantiated from its config — no pretrained weights
    are downloaded.  Tokenizers are loaded normally (they are lightweight
    and have no model weights).  Other non-module components (schedulers,
    feature extractors) are set to ``None``.

    This is called inside ``init_empty_weights()`` by
    :meth:`DiffusionModel._load_meta`, so all created parameters land
    on the ``meta`` device.

    Args:
        automodel: The pipeline class (e.g. ``DiffusionPipeline``).
        repo_id: HuggingFace repository ID.
        revision: Git revision / branch / tag.
        **kwargs: User overrides (e.g. ``safety_checker=None``).

    Returns:
        A pipeline instance with meta-device module components.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    config = automodel.load_config(repo_id, revision=revision)

    pipe_cls_name = config.get("_class_name", automodel.__name__)
    pipe_cls = getattr(pipelines, pipe_cls_name, automodel)

    components = {}
    for key, val in config.items():
        if key.startswith("_"):
            continue

        if not isinstance(val, list):
            kwargs[key] = val
            continue

        lib_name, cls_name = val

        # Honour explicit user overrides (e.g. safety_checker=None)
        if key in kwargs:
            components[key] = kwargs.pop(key)
            continue

        cls = _resolve_component_cls(lib_name, cls_name)

        if cls is None:
            components[key] = None
            continue

        if isinstance(cls, type) and issubclass(cls, PreTrainedTokenizerBase):
            try:
                components[key] = cls.from_pretrained(
                    repo_id, subfolder=key, revision=revision
                )
            except Exception:
                components[key] = None
            continue

        if not (isinstance(cls, type) and issubclass(cls, torch.nn.Module)):
            components[key] = None
            continue

        # Create meta-device component from its config
        try:
            if hasattr(cls, "load_config"):
                # Diffusers components (UNet, VAE, Transformer, etc.)
                sub_cfg = cls.load_config(repo_id, subfolder=key, revision=revision)
                with init_empty_weights():
                    components[key] = cls.from_config(sub_cfg)
            else:
                # Transformers components (CLIPTextModel, T5, etc.)
                auto_cfg = AutoConfig.from_pretrained(
                    repo_id, subfolder=key, revision=revision
                )
                with init_empty_weights():
                    components[key] = cls(auto_cfg)
        except Exception:
            components[key] = None

    # Filter out from_pretrained()-specific kwargs that the pipeline
    # constructor doesn't accept (e.g. torch_dtype, variant, device_map).
    init_sig = inspect.signature(pipe_cls.__init__)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in init_sig.parameters.values()
    )
    if not has_var_keyword:
        valid_params = set(init_sig.parameters.keys()) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return pipe_cls(**components, **kwargs)


class Diffuser(util.WrapperModule):
    """Wrapper module that loads a diffusion pipeline and exposes its components as submodules.

    All pipeline components that are ``torch.nn.Module`` or
    ``PreTrainedTokenizerBase`` instances are registered as attributes
    so they appear in the Envoy tree and can be traced. The exact
    component names depend on the pipeline (e.g. ``unet`` for Stable
    Diffusion, ``transformer`` for Flux, plus ``vae``, ``text_encoder``,
    etc.).

    Can be constructed in two ways:

    1. **From pretrained** (default): pass a pipeline class and repo ID
       to download and load full weights via ``from_pretrained()``.
    2. **From a pre-built pipeline**: pass a ``DiffusionPipeline``
       instance directly (used by :meth:`DiffusionModel._load_meta`
       for meta-tensor initialization).

    Args:
        automodel_or_pipeline: Either a pipeline class
            (``Type[DiffusionPipeline]``) for ``from_pretrained`` loading,
            or an already-constructed ``DiffusionPipeline`` instance.
        *args: Forwarded to ``automodel.from_pretrained()`` when loading.
        **kwargs: Forwarded to ``automodel.from_pretrained()`` when loading.

    Attributes:
        pipeline (DiffusionPipeline): The underlying diffusers pipeline.
    """

    def __init__(self, automodel_or_pipeline=DiffusionPipeline, *args, **kwargs) -> None:
        super().__init__()

        if isinstance(automodel_or_pipeline, DiffusionPipeline):
            self.pipeline = automodel_or_pipeline
        else:
            self.pipeline = automodel_or_pipeline.from_pretrained(*args, **kwargs)

        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module) or isinstance(
                value, PreTrainedTokenizerBase
            ):
                setattr(self, key, value)

    def generate(self, *args, **kwargs):
        """Run the full diffusion pipeline.

        Calls the pipeline's ``__call__`` method (not ``.generate()``,
        which does not exist on ``DiffusionPipeline``).

        Returns:
            The pipeline output (typically a dataclass with ``.images``).
        """
        return self.pipeline(*args, **kwargs)


class DiffusionModel(HuggingFaceModel):
    """NNsight wrapper for diffusion pipelines.

    Wraps any ``diffusers.DiffusionPipeline`` so that its components
    can be traced and intervened on. Works with UNet-based pipelines
    (Stable Diffusion) and transformer-based pipelines (Flux, DiT)
    alike — the denoiser is accessible as whatever attribute the
    pipeline exposes (``model.unet`` or ``model.transformer``).

    By default, ``.trace()`` runs the full diffusion pipeline with
    ``num_inference_steps=1`` for fast single-step tracing. Use
    ``.generate()`` to run the full pipeline with the default or
    user-specified number of inference steps.

    When ``dispatch=False`` (the default), only lightweight config
    files are downloaded and the model architecture is created with
    meta tensors (no memory).  Real weights are loaded automatically
    on the first ``.trace()`` or ``.generate()`` call, or explicitly
    via ``.dispatch()``.

    Examples::

        # Stable Diffusion (UNet-based)
        sd = DiffusionModel("stabilityai/stable-diffusion-2-1")
        with sd.generate("A cat", num_inference_steps=50) as tracer:
            for step in tracer.iter[:]:
                denoiser_out = sd.unet.output.save()

        # Flux (Transformer-based)
        flux = DiffusionModel("black-forest-labs/FLUX.1-schnell")
        with flux.trace("A cat"):
            denoiser_out = flux.transformer.output.save()

    Args:
        *args: Forwarded to :class:`HuggingFaceModel`.  The first
            positional argument is typically a repo ID string.
        automodel (Type[DiffusionPipeline]): The diffusers pipeline
            class (or a string name resolvable from ``diffusers.pipelines``).
            Defaults to ``DiffusionPipeline``.
        **kwargs: Forwarded to the pipeline's ``from_pretrained()``.

    Attributes:
        automodel (Type[DiffusionPipeline]): The pipeline class used for loading.
    """

    def __init__(
        self, *args, automodel: Type[DiffusionPipeline] = DiffusionPipeline, **kwargs
    ) -> None:

        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(pipelines, automodel)
        )

        self._model: Diffuser = None

        super().__init__(*args, **kwargs)

    def _load_meta(self, repo_id: str, revision: Optional[str] = None, **kwargs):
        """Load a meta (placeholder) version of the diffusion model.

        Downloads only the pipeline and component config files (a few KB).
        Each ``nn.Module`` component is instantiated from its config with
        meta-device parameters — no pretrained weights are downloaded.
        Tokenizers are loaded normally (they are lightweight).  Other
        non-module components (scheduler, etc.) are set to ``None`` and
        will be loaded on :meth:`dispatch`.

        Args:
            repo_id: HuggingFace repository ID.
            revision: Git revision of the repository.
            **kwargs: User overrides forwarded to pipeline construction
                (e.g. ``safety_checker=None``).

        Returns:
            A :class:`Diffuser` instance with meta-device parameters.
        """
        pipeline = _build_pipeline_from_config(
            self.automodel, repo_id, revision=revision, **kwargs
        )

        return Diffuser(pipeline)

    def _load(
        self, repo_id: str, revision: Optional[str] = None, device_map=None, **kwargs
    ) -> Diffuser:
        """Load the diffusion model with full weights.

        Args:
            repo_id: HuggingFace repository ID.
            revision: Git revision of the repository.
            device_map: Device placement strategy.
            **kwargs: Forwarded to ``Diffuser()``.

        Returns:
            A :class:`Diffuser` instance.
        """

        device_map = "balanced" if device_map == "auto" or device_map is None else device_map

        model = Diffuser(
            self.automodel, repo_id, revision=revision, device_map=device_map, **kwargs
        )

        return model

    def _prepare_input(self, *inputs, **kwargs):
        """Normalize raw user input into a consistent format for batching.

        Accepts a single string prompt or a list of string prompts.
        Returns ``(args, kwargs, batch_size)`` where args is a tuple
        containing the prompt list.

        Args:
            *inputs: A single string or list of strings.
            **kwargs: Additional keyword arguments (passed through).

        Returns:
            Tuple of ``((prompts,), kwargs, batch_size)``.
        """
        if len(inputs) == 0:
            return tuple(), kwargs, 0

        assert len(inputs) == 1
        prompt = inputs[0]

        if isinstance(prompt, str):
            prompt = [prompt]

        return (prompt,), kwargs, len(prompt)

    def _batch(self, batched_input, *args, **kwargs):
        """Combine a new invoke's prepared prompts with already-batched prompts.

        Merges prompt lists from multiple invokes into a single list
        for batched pipeline execution.

        Args:
            batched_input: A tuple of ``(batched_args, batched_kwargs)``
                from all previous invokes.
            *args: The new invoke's prepared positional arguments.
            **kwargs: The new invoke's prepared keyword arguments.

        Returns:
            Tuple of ``(combined_args, combined_kwargs)``.
        """
        batched_args, batched_kwargs = batched_input

        if len(args) > 0:
            combined_prompts = list(batched_args[0]) + list(args[0])
        else:
            combined_prompts = list(batched_args[0])

        combined_kwargs = {**batched_kwargs, **kwargs}

        return (combined_prompts,), combined_kwargs

    def _run_pipeline(self, prepared_inputs, *args, seed=None, **kwargs):
        """Shared pipeline execution logic for both trace and generate.

        Sets up iteration step counting on the interleaver, handles
        seed/generator creation, runs the pipeline, wraps the output
        through the model's forward (for hook access), and resets
        ``default_all`` afterward.

        Args:
            prepared_inputs: The prompt list from ``_prepare_input``.
            *args: Additional positional arguments for the pipeline.
            seed: Random seed for reproducibility. If provided with
                multiple prompts, each prompt gets ``seed + offset``.
            **kwargs: Keyword arguments forwarded to the pipeline
                (e.g. ``num_inference_steps``, ``guidance_scale``).

        Returns:
            The pipeline output passed through the wrapper module.
        """
        if self._interleaver is not None:
            steps = kwargs.get("num_inference_steps")
            if steps is None:
                try:
                    steps = (
                        inspect.signature(self._model.pipeline.__call__)
                        .parameters["num_inference_steps"]
                        .default
                    )
                except Exception:
                    steps = 50
            self._interleaver.default_all = steps

        generator = torch.Generator(self.device)

        if seed is not None:
            if isinstance(prepared_inputs, list) and len(prepared_inputs) > 1:
                generator = [
                    torch.Generator(self.device).manual_seed(seed + offset)
                    for offset in range(
                        len(prepared_inputs) * kwargs.get("num_images_per_prompt", 1)
                    )
                ]
            else:
                generator = generator.manual_seed(seed)

        output = self._model.pipeline(
            prepared_inputs, *args, generator=generator, **kwargs
        )

        if self._interleaver is not None:
            self._interleaver.default_all = None

        output = self._model(output)

        return output

    def __call__(self, prepared_inputs, *args, **kwargs):
        """Run the full diffusion pipeline with a 1-step default.

        Used by ``.trace()`` — defaults to ``num_inference_steps=1``
        for fast single-step tracing unless the user overrides it.

        Args:
            prepared_inputs: The prompt list from ``_prepare_input``.
            *args: Additional positional arguments for the pipeline.
            **kwargs: Keyword arguments forwarded to the pipeline.

        Returns:
            The pipeline output passed through the wrapper module.
        """
        kwargs.setdefault("num_inference_steps", 1)
        return self._run_pipeline(prepared_inputs, *args, **kwargs)

    def __nnsight_generate__(self, prepared_inputs, *args, **kwargs):
        """Run the full diffusion pipeline for ``.generate()`` contexts.

        Unlike ``__call__``, this does not set a default for
        ``num_inference_steps``, allowing the pipeline's own default
        (or the user's explicit value) to take effect.

        Args:
            prepared_inputs: The prompt list from ``_prepare_input``.
            *args: Additional positional arguments for the pipeline.
            **kwargs: Keyword arguments forwarded to the pipeline.

        Returns:
            The pipeline output passed through the wrapper module.
        """
        return self._run_pipeline(prepared_inputs, *args, **kwargs)

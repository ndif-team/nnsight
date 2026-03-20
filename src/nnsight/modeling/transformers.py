from .huggingface import HuggingFaceModel

from torch.nn.modules import Module
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from typing import Optional
from typing import Type
from transformers.models.auto import modeling_auto
from transformers import AutoModel


class TransformersModel(HuggingFaceModel):
    """NNsight wrapper for HuggingFace Transformers models.

    Adds ``AutoConfig`` / ``AutoModel`` support on top of
    :class:`HuggingFaceModel`. Handles config loading, meta-tensor
    initialization via ``from_config``, and full weight loading via
    ``from_pretrained``.

    Args:
        *args: Forwarded to :class:`HuggingFaceModel`.  The first
            positional argument is typically a repo ID string or a
            pre-loaded ``torch.nn.Module``.
        config_model (Optional[Type[PretrainedConfig]]): An explicit
            HuggingFace config instance to use instead of loading one
            from the repo. Defaults to ``None`` (auto-loaded).
        automodel (Type[AutoModel]): The ``AutoModel`` class to use for
            loading (e.g. ``AutoModelForCausalLM``).
            Defaults to ``AutoModel``.
        **kwargs: Forwarded to ``from_pretrained`` / ``from_config``.

    Attributes:
        config (PretrainedConfig): The model's HuggingFace configuration.
        automodel (Type[AutoModel]): The ``AutoModel`` class used for loading.
    """

    def __init__(self, *args, config_model: Type[PretrainedConfig] = None, automodel: Type[AutoModel] = AutoModel, **kwargs):

        self.config: PretrainedConfig = config_model
        
        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(modeling_auto, automodel)
        )

        super().__init__(*args, **kwargs)

    def _load_config(self, repo_id: str, revision: Optional[str] = None, **kwargs):

        if self.config is None:

            self.config = AutoConfig.from_pretrained(
                repo_id, revision=revision, **kwargs
            )

    def _load_meta(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        **kwargs,
    ) -> Module:

        self._load_config(repo_id, revision=revision, **kwargs)

        model = self.automodel.from_config(self.config, trust_remote_code=True)

        self.config = model.config

        return model

    def _load(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        **kwargs,
    ) -> PreTrainedModel:

        load_format = kwargs.pop("load_format", None)
        gpu_direct = kwargs.pop("gpu_direct", True)
        concurrency = kwargs.pop("concurrency", 16)
        self._load_config(repo_id, revision=revision, **kwargs)

        # Tensor parallelism requires torch.distributed, which conflicts with
        # RunAI's DistributedStreamer.  Use from_pretrained for TP loads.
        if kwargs.get("tp_plan") is not None:
            load_format = "from_pretrained"

        # Default: try run:ai streamer, fall back to from_pretrained if not installed
        if load_format != "from_pretrained":
            try:
                model = self._load_streamed(
                    repo_id, revision=revision,
                    gpu_direct=gpu_direct, concurrency=concurrency,
                    **kwargs,
                )
                self.config = model.config
                return model
            except ImportError:
                if load_format == "runai_streamer":
                    raise  # explicit request — don't swallow the error
                # else load_format is None (default) — fall through silently

        model = self.automodel.from_pretrained(repo_id, revision=revision, **kwargs)

        self.config = model.config

        return model

    def _resolve_device_map(self, device_map_str: str, max_memory=None,
                            torch_dtype=None) -> dict:
        """Expand a string device_map ('auto', 'balanced', etc.) to a dict.

        Creates a throwaway meta-device model to compute the layer→device
        mapping, then discards it.  This lets us know each tensor's target
        GPU *before* streaming begins.

        ``torch_dtype`` sets the default dtype during meta-model creation
        so that memory estimates match the actual loading dtype (e.g.
        bfloat16 vs float32).

        When the model config contains a ``quantization_config``, the
        corresponding ``HfQuantizer`` is created and its
        ``preprocess_model`` is called to replace modules with quantized
        variants (e.g. ``Mxfp4GptOssExperts``).  This ensures
        ``compute_module_sizes`` sees the smaller quantized parameter
        shapes rather than overestimating at full-precision sizes.
        """
        from transformers.integrations.accelerate import _get_device_map
        import torch

        # Build quantizer so device-map sizing accounts for quantized weights
        hf_quantizer = None
        if getattr(self.config, "quantization_config", None) is not None:
            try:
                from transformers.quantizers.auto import AutoHfQuantizer
                hf_quantizer = AutoHfQuantizer.from_config(
                    self.config.quantization_config, pre_quantized=True,
                )
            except Exception:
                pass  # Missing deps — fall back to unquantized sizing

        model_class = self.automodel._model_mapping[type(self.config)]
        old_dtype = torch.get_default_dtype()
        effective_dtype = torch_dtype or getattr(self.config, "torch_dtype", None)
        if effective_dtype is not None:
            torch.set_default_dtype(effective_dtype)
        try:
            with torch.device("meta"):
                meta_model = model_class(self.config)
            # Run quantizer's module replacement (e.g. GptOssExperts →
            # Mxfp4GptOssExperts) so that compute_module_sizes sees the
            # smaller quantized parameter shapes.  This mirrors what
            # from_pretrained does before calling _get_device_map.
            if hf_quantizer is not None:
                try:
                    hf_quantizer.preprocess_model(
                        meta_model, dtype=effective_dtype,
                    )
                except Exception:
                    pass  # Non-fatal — sizes will be overestimated
            resolved = _get_device_map(
                meta_model, device_map_str, max_memory, hf_quantizer=hf_quantizer,
            )
        finally:
            torch.set_default_dtype(old_dtype)
        del meta_model
        return resolved

    def _load_streamed(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        concurrency: int = 16,
        gpu_direct: bool = True,
        **kwargs,
    ) -> PreTrainedModel:
        """Load model using run:ai SafetensorsStreamer for fast disk I/O.

        Builds a lazy state dict whose values stream incrementally on
        first ``__getitem__`` access — peak CPU memory is ~1 tensor
        instead of the full model.

        When *gpu_direct* is True (default) and ``device_map`` is a
        string like ``"auto"``, it is resolved to a concrete dict
        *before* building the lazy state dict so that the streaming
        cache can copy tensors directly to their target GPU, making
        HF's ``_materialize_copy().to()`` a no-op.

        When *gpu_direct* is False, tensors are cloned to CPU and HF
        workers handle the GPU transfer (the pre-GPU-direct path).

        ``from_pretrained(None, state_dict=...)`` handles weight
        renaming, conversion, dtype casting, device placement, and
        tied weight resolution.
        """
        from .loader import (
            resolve_shard_paths,
            build_lazy_state_dict,
        )

        shard_paths = resolve_shard_paths(repo_id, revision=revision)

        # Resolve device_map early so the cache can place tensors on GPU
        device_map = kwargs.pop("device_map", None)
        resolved_device_map = None

        if gpu_direct:
            # Resolve device_map early so the cache can place tensors on GPU
            if isinstance(device_map, str) and device_map in (
                "auto", "balanced", "balanced_low_0", "sequential",
            ):
                resolved_device_map = self._resolve_device_map(
                    device_map, max_memory=kwargs.get("max_memory"),
                    torch_dtype=kwargs.get("torch_dtype"),
                )
            elif isinstance(device_map, dict):
                resolved_device_map = device_map

        state_dict = build_lazy_state_dict(
            shard_paths, concurrency=concurrency,
            device_map=resolved_device_map,
            torch_dtype=kwargs.get("torch_dtype") if gpu_direct else None,
        )

        # Resolve concrete model class — Auto classes reject None as path
        model_class = self.automodel._model_mapping[type(self.config)]

        model = model_class.from_pretrained(
            None,
            config=self.config,
            state_dict=state_dict,
            revision=revision,
            device_map=resolved_device_map or device_map,
            **kwargs,
        )
        return model

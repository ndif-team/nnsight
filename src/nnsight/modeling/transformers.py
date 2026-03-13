from .huggingface import HuggingFaceModel

from torch.nn.modules import Module
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from typing import Dict, Optional, Union
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
        self._load_config(repo_id, revision=revision, **kwargs)

        # Default: try run:ai streamer, fall back to from_pretrained if not installed
        if load_format != "from_pretrained":
            try:
                model = self._load_streamed(repo_id, revision=revision, **kwargs)
                self.config = model.config
                return model
            except ImportError:
                if load_format == "runai_streamer":
                    raise  # explicit request — don't swallow the error
                # else load_format is None (default) — fall through silently

        model = self.automodel.from_pretrained(repo_id, revision=revision, **kwargs)

        self.config = model.config

        return model

    def _load_streamed(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        device_map: Union[str, Dict[str, str]] = "auto",
        torch_dtype=None,
        max_memory: Optional[Dict] = None,
        trust_remote_code: bool = False,
        concurrency: int = 16,
        **kwargs,
    ) -> PreTrainedModel:
        """Load model using run:ai SafetensorsStreamer for fast pipelined weight loading.

        Creates an empty model from config, computes device placement, then
        streams weights from safetensors shards with concurrent I/O and
        pipelined pinned-memory GPU transfers.
        """
        # Import early so ImportError propagates to _load() before doing work
        from runai_model_streamer import SafetensorsStreamer  # noqa: F401
        from accelerate import init_empty_weights, infer_auto_device_map
        from .loader import resolve_shard_paths, stream_weights_into_model

        # 1. Create empty model from config (self.config already loaded by _load_config)
        with init_empty_weights():
            model = self.automodel.from_config(
                self.config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=kwargs.get("attn_implementation"),
            )

        # 2. Resolve device_map
        _DEVICE_MAP_STRATEGIES = {"auto", "balanced", "balanced_low_0", "sequential"}
        if isinstance(device_map, str):
            if device_map in _DEVICE_MAP_STRATEGIES:
                model.tie_weights()
                no_split = (
                    getattr(self.config, "_no_split_modules", None)
                    or getattr(model, "_no_split_module_classes", None)
                )
                if max_memory is None:
                    from accelerate.utils import get_max_memory
                    max_memory = get_max_memory()
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split,
                    dtype=torch_dtype,
                )
            else:
                # device_map is a device name like "cuda:0" or "cpu"
                device_map = {"": device_map}

        # 3. Stream weights
        shard_paths = resolve_shard_paths(repo_id, revision=revision)
        stream_weights_into_model(
            model,
            shard_paths,
            device_map,
            torch_dtype=torch_dtype,
            concurrency=concurrency,
        )

        model.tie_weights()
        model.eval()
        return model

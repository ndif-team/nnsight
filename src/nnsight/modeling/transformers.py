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
        pin_memory = kwargs.pop("pin_memory", False)
        self._load_config(repo_id, revision=revision, **kwargs)

        # Default: try run:ai streamer, fall back to from_pretrained if not installed
        if load_format != "from_pretrained":
            try:
                model = self._load_streamed(
                    repo_id, revision=revision,
                    pin_memory=pin_memory, **kwargs,
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

    def _load_streamed(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        concurrency: int = 16,
        pin_memory: bool = False,
        **kwargs,
    ) -> PreTrainedModel:
        """Load model using run:ai SafetensorsStreamer for fast disk I/O.

        Builds a lazy state dict whose values stream shard-by-shard on
        first ``__getitem__`` access — peak CPU memory is ~2-3 shards
        instead of the full model.

        ``from_pretrained(None, state_dict=...)`` handles weight renaming,
        conversion, dtype casting, device placement, and tied weight
        resolution.
        """
        from .loader import (
            resolve_shard_paths,
            build_lazy_state_dict,
        )

        shard_paths = resolve_shard_paths(repo_id, revision=revision)
        state_dict = build_lazy_state_dict(shard_paths, concurrency=concurrency, pin_memory=pin_memory)

        # Resolve concrete model class — Auto classes reject None as path
        model_class = self.automodel._model_mapping[type(self.config)]

        model = model_class.from_pretrained(
            None,
            config=self.config,
            state_dict=state_dict,
            revision=revision,
            **kwargs,
        )
        return model

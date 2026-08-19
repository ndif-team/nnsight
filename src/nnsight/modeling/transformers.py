from .huggingface import HuggingFaceModel

from torch.nn.modules import Module
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from typing import Optional
from typing import Type
from transformers.models.auto import modeling_auto
from transformers import AutoModel


def _load_peft_adapter(model, adapter_id: str, **kwargs):
    """Wrap ``model`` with a PEFT adapter, if the ``peft`` package is installed.

    Imported lazily so ``peft`` stays an optional dependency — nnsight
    users who never pass ``adapter_id`` should not need it installed.
    """
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            "adapter_id was passed but the `peft` package is not installed. "
            "Install it with `pip install peft`."
        ) from e

    return PeftModel.from_pretrained(model, adapter_id, **kwargs)


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

    def __init__(
        self,
        *args,
        config_model: Type[PretrainedConfig] = None,
        automodel: Type[AutoModel] = AutoModel,
        **kwargs,
    ):

        # Use __dict__ directly so we don't mirror this onto the (possibly
        # already-loaded) underlying module via Envoy.__setattr__ — we're
        # caching the config on the wrapper, not mutating the model's own.
        self.__dict__["config"] = config_model

        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(modeling_auto, automodel)
        )

        super().__init__(*args, **kwargs)

    def _load_config(self, repo_id: str, revision: Optional[str] = None, **kwargs):

        if self.config is None:

            self.__dict__["config"] = AutoConfig.from_pretrained(
                repo_id, revision=revision, **kwargs
            )

    def _load_meta(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        adapter_id: Optional[str] = None,
        **kwargs,
    ) -> Module:

        self._load_config(repo_id, revision=revision, **kwargs)

        model = self.automodel.from_config(
            self.config, trust_remote_code=kwargs.get("trust_remote_code", False)
        )

        self.__dict__["config"] = model.config

        # adapter_id is intentionally not applied here: this method builds a
        # meta-tensor skeleton, and PEFT's from_pretrained needs to read real
        # adapter weight files onto real storage. The adapter is loaded later
        # in _load(), once dispatch() has replaced meta tensors with real ones.
        return model

    def _load(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        adapter_id: Optional[str] = None,
        **kwargs,
    ) -> PreTrainedModel:

        self._load_config(repo_id, revision=revision, **kwargs)

        model = self.automodel.from_pretrained(repo_id, revision=revision, **kwargs)

        self.__dict__["config"] = model.config

        if adapter_id is not None:
            model = _load_peft_adapter(model, adapter_id)

        return model

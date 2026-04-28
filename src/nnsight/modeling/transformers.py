from peft import PeftModel
from .huggingface import HuggingFaceModel

from torch.nn.modules import Module
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from typing import Any, Dict, Optional
from typing import Type
from transformers.models.auto import modeling_auto
from transformers import AutoModel
import warnings


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
        peft (Optional[str]): HuggingFace repo id of a PEFT adapter to
            apply during remote execution. Forwarded to NDIF via the
            ``ndif-extras`` header; a PEFT-aware server actor wraps the
            base model with ``PeftModel.from_pretrained`` for the request
            and unloads the adapter on cleanup. Defaults to ``None``
            (no adapter).
        **kwargs: Forwarded to ``from_pretrained`` / ``from_config``.

    Attributes:
        config (PretrainedConfig): The model's HuggingFace configuration.
        automodel (Type[AutoModel]): The ``AutoModel`` class used for loading.
        peft (Optional[str]): PEFT adapter repo id, if any.
    """

    def __init__(
        self,
        *args,
        config_model: Type[PretrainedConfig] = None,
        automodel: Type[AutoModel] = AutoModel,
        peft: Optional[str] = None,
        **kwargs,
    ):

        self.config: PretrainedConfig = config_model

        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(modeling_auto, automodel)
        )

        self.peft = peft

        super().__init__(*args, **kwargs)

    def _remoteable_extras(self) -> Dict[str, Any]:
        if self.peft is None:
            return {}
        return {"peft_repo_id": self.peft}

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

        if self.peft is not None:

            warnings.filterwarnings("ignore", category=UserWarning)

            model = PeftModel.from_pretrained(model, self.peft)

            warnings.filterwarnings("default", category=UserWarning)

        return model

    def _load(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        **kwargs,
    ) -> PreTrainedModel:

        self._load_config(repo_id, revision=revision, **kwargs)

        model = self.automodel.from_pretrained(repo_id, revision=revision, **kwargs)

        self.config = model.config

        if self.peft is not None:
            warnings.filterwarnings("ignore", category=UserWarning)

            model = PeftModel.from_pretrained(model, self.peft)

            warnings.filterwarnings("default", category=UserWarning)

        return model

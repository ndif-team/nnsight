from peft import PeftModel
from .huggingface import HuggingFaceModel
from ..intervention.envoy import Envoy

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
            ``ndif-env`` header; the base ``ModelActor`` calls
            :meth:`_remoteable_set_env`, which wraps the base model with
            ``PeftModel.from_pretrained`` for the request and reuses the
            loaded adapter across requests, swapping only when the
            requested repo id changes. Defaults to ``None`` (no adapter).
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

        # Use __dict__ directly so we don't mirror this onto the (possibly
        # already-loaded) underlying module via Envoy.__setattr__ — we're
        # caching the config on the wrapper, not mutating the model's own.
        self.__dict__["config"] = config_model

        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(modeling_auto, automodel)
        )

        self.peft = peft

        super().__init__(*args, **kwargs)

    def _remoteable_get_env(self) -> Dict[str, Any]:
        if self.peft is None:
            return {}
        return {"peft_repo_id": self.peft}

    def _remoteable_set_env(self, env: Dict[str, Any]) -> None:
        """Swap the PEFT adapter to match ``env["peft_repo_id"]``.

        Compares the requested repo id against ``self.peft`` and only
        rewraps the underlying module when they differ. Transitions:

            current  requested  action
            -------  ---------  ------
            None     None       no-op
            None     X          load X
            X        X          no-op
            X        Y          unload X, load Y
            X        None       unload X
        """

        requested = env.get("peft_repo_id") if env else None

        if requested == self.peft:
            return

        if self.peft is not None:
            Envoy.__init__(
                self, self._module.unload(), interleaver=self.interleaver
            )

        if requested:
            peft_model = PeftModel.from_pretrained(self._module, requested)
            Envoy.__init__(self, peft_model, interleaver=self.interleaver)

        self.peft = requested

    def _load_config(self, repo_id: str, revision: Optional[str] = None, **kwargs):

        if self.config is None:

            self.__dict__["config"] = AutoConfig.from_pretrained(
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

        self.__dict__["config"] = model.config

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

        self.__dict__["config"] = model.config

        if self.peft is not None:
            warnings.filterwarnings("ignore", category=UserWarning)

            model = PeftModel.from_pretrained(model, self.peft)

            warnings.filterwarnings("default", category=UserWarning)

        return model

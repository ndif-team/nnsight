from .huggingface import HuggingFaceModel
from ._kernel_shim import meta_kernel_shim

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

            # Default to trusting remote code so the config class matches the
            # remote modeling code used for meta/dispatch. Some remote configs
            # understand fields the native class does not (e.g. newer Nemotron-H
            # ``hybrid_override_pattern`` block types).
            kwargs.setdefault("trust_remote_code", True)

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

        # Keep the meta implementation consistent with the dispatched one: both
        # default to trusting remote code so the intervention tree the client
        # builds matches the model that is actually loaded/served (e.g. the
        # Nemotron-H remote code, whose per-expert layout differs from the native
        # transformers class).
        trust_remote_code = kwargs.get("trust_remote_code", True)

        # Some remote modeling files hard-import CUDA-only kernels (mamba_ssm,
        # causal_conv1d) at module import time. A meta model never runs a forward,
        # so satisfy those imports with inert stubs and keep the client GPU-free.
        with meta_kernel_shim():
            model = self.automodel.from_config(
                self.config, trust_remote_code=trust_remote_code
            )

        self.__dict__["config"] = model.config

        return model

    def _load(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        **kwargs,
    ) -> PreTrainedModel:

        self._load_config(repo_id, revision=revision, **kwargs)

        # Mirror the meta path's default so dispatch loads the same implementation
        # the intervention tree was built against.
        kwargs.setdefault("trust_remote_code", True)

        model = self.automodel.from_pretrained(repo_id, revision=revision, **kwargs)

        self.__dict__["config"] = model.config

        return model

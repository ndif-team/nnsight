from __future__ import annotations

import json
from typing import Any, Optional

import torch

from .mixins.remotable import Remotable

# Canonical repo ids, keyed by the user-supplied id (resolves casing/redirects
# so different spellings of the same model produce the same remote key).
_ID_CACHE: dict[str, str] = {}


class HuggingFaceModel(Remotable):
    """nnsight wrapper around a HuggingFace Hub model.

    Builds the architecture on the meta device from the repo's config and loads
    real weights on dispatch, both via transformers. The transformers auto class
    is configurable through ``AUTO_CLASS`` so subclasses can target, e.g.,
    ``AutoModelForCausalLM``.

    Args:
        repo_id: A HuggingFace repo id (e.g. "openai-community/gpt2") or an
            already-loaded torch.nn.Module.
        revision: Optional git revision (branch/tag/commit) of the repo.
    """

    AUTO_CLASS = "AutoModel"

    def __init__(
        self,
        repo_id: Any,
        *args: Any,
        revision: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.repo_id = (
            repo_id if isinstance(repo_id, str) else getattr(repo_id, "name_or_path", None)
        )
        self.revision = revision
        # revision is used by _load/_load_meta via self.revision, not threaded
        # through super (it must not reach Envoy.__init__ on the passthrough).
        super().__init__(repo_id, *args, **kwargs)

    def _auto(self) -> Any:
        import transformers  # lazy: transformers is heavy and only needed to load

        return getattr(transformers, self.AUTO_CLASS)

    def _load_meta(self, repo_id: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(repo_id, revision=self.revision)
        return self._auto().from_config(config)

    def _load(self, repo_id: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
        return self._auto().from_pretrained(repo_id, revision=self.revision, **kwargs)

    def _remoteable_model_key(self) -> str:
        # Canonicalize the repo id via the Hub (cached) so the server resolves
        # the same model regardless of how the id was spelled.
        if self.repo_id not in _ID_CACHE:
            from huggingface_hub import HfApi

            _ID_CACHE[self.repo_id] = HfApi().model_info(self.repo_id).id

        return json.dumps(
            {"repo_id": _ID_CACHE[self.repo_id], "revision": self.revision}
        )

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs: Any) -> HuggingFaceModel:
        data = {**json.loads(model_key), **kwargs}
        repo_id = data.pop("repo_id")
        revision = data.pop("revision", None)
        return cls(repo_id, revision=revision, **data)

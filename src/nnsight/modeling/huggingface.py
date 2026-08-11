from __future__ import annotations

import json
import logging
import math
from typing import Any, Optional

import torch

from .mixins.remotable import Remotable

logger = logging.getLogger("nnsight")

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

        # A model loaded with `distributed_config=DistributedConfig(tp_size=N)`
        # has its linears split across ranks, so a plain interleaver would hand
        # intervention code one rank's slice of an activation. Whether this model
        # is sharded isn't known yet — it only becomes visible as the weights
        # load — so give the tree an interleaver that can handle it either way:
        # it stays disabled (and behaves exactly like the base) unless it finds
        # something actually split while instrumenting. Explicit rather than
        # setdefault so a caller-supplied interleaver still wins.
        if "interleaver" not in kwargs:
            from .tp import TPInterleaver

            kwargs["interleaver"] = TPInterleaver()

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
    def _remoteable_estimate_bytes(
        cls, model_key: str, dtype: str, trust_remote_code: bool = False
    ) -> int:
        """Size the weights from the Hub's parameter count, without building the
        architecture.

        The Hub indexes every safetensors checkpoint and reports how many
        parameters it holds, so the number the base class builds a whole meta
        model to count is one request away. Falls back to that build when the
        count isn't published — an older ``.bin``-only repo, a private mirror, or
        no network — so this is a shortcut, never the only route.

        Buffers are not in the Hub's count (they aren't checkpoint tensors).
        They're a rounding error next to the parameters for a transformer, and
        the caller pads.
        """
        from .mixins.remotable import bytes_per_element

        data = json.loads(model_key)
        repo_id, revision = data.get("repo_id"), data.get("revision")

        parameters = cls._hub_parameter_count(repo_id, revision)
        if parameters is None:
            logger.debug(
                f"No parameter count published for {repo_id!r}; sizing it by "
                "building the architecture instead"
            )
            return super()._remoteable_estimate_bytes(
                model_key, dtype, trust_remote_code=trust_remote_code
            )

        return math.ceil(parameters * bytes_per_element(dtype))

    @staticmethod
    def _hub_parameter_count(repo_id: str, revision: Optional[str]) -> Optional[int]:
        """Total parameters the Hub reports for a repo, or None if it doesn't."""
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(repo_id, revision=revision)
        except Exception:
            return None

        safetensors = getattr(info, "safetensors", None)
        return getattr(safetensors, "total", None)

    @classmethod
    def _remoteable_checkpoint_config(
        cls, model_key: str, trust_remote_code: bool = False
    ) -> Optional[Any]:
        """The repo's config, read from the Hub cache without any weights."""
        return cls._config(model_key, trust_remote_code)

    @classmethod
    def _config(cls, model_key: str, trust_remote_code: bool) -> Optional[Any]:
        from transformers import AutoConfig

        data = json.loads(model_key)
        try:
            return AutoConfig.from_pretrained(
                data.get("repo_id"),
                revision=data.get("revision"),
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            logger.debug(f"Could not read a config for {data.get('repo_id')!r}")
            return None

    @classmethod
    def _remoteable_checkpoint_revision(
        cls, model_key: str, **kwargs: Any
    ) -> Optional[str]:
        """The revision the key pins, straight out of it."""
        return json.loads(model_key).get("revision")

    @classmethod
    def _remoteable_max_tp_size(
        cls, model_key: str, trust_remote_code: bool = False
    ) -> Optional[int]:
        """The largest tensor-parallel degree, read from the checkpoint's config.

        Config only — no weights, no architecture — since the sharding plan and
        the dimensions it has to divide are both declared there. See
        [`max_tp_size`][nnsight.modeling.tp.plan.max_tp_size].
        """
        from .tp import max_tp_size

        config = cls._config(model_key, trust_remote_code)
        return max_tp_size(config) if config is not None else None

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs: Any) -> HuggingFaceModel:
        data = {**json.loads(model_key), **kwargs}
        repo_id = data.pop("repo_id")
        revision = data.pop("revision", None)
        return cls(repo_id, revision=revision, **data)

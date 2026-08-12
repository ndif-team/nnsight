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

# Configs already read, keyed by (model_key, trust_remote_code) — see `_config`.
_CONFIG_CACHE: dict[tuple, Any] = {}

# How long to wait on the Hub when describing a checkpoint. These calls happen on
# a placement path — a server deciding where a model goes, before it loads
# anything — where the caller would rather be told "couldn't reach the Hub" than
# wait indefinitely.
#
# Per request, matching what huggingface_hub already applies to everything it
# fetches (`HF_HUB_ETAG_TIMEOUT` and `HF_HUB_DOWNLOAD_TIMEOUT`, both 10s). So
# `AutoConfig.from_pretrained` is bounded whether or not anything here says so —
# it takes no timeout argument, but its transport does — and the number below
# only has to be passed where an API accepts one. What is *not* bounded is the
# number of requests a read makes, so treat this as a bound on progress, not on
# the call.
HUB_TIMEOUT_SECONDS = 10.0


class CheckpointUnreachable(Exception):
    """The checkpoint could not be read — the Hub was slow, down, or refused.

    Distinct from *reading it and finding nothing*, which every reader here
    reports as ``None``. Collapsing the two is how a network problem comes to
    look like a fact about a model: a config that failed to download makes
    ``max_tp_size`` return ``None``, and ``None`` is the answer meaning "this
    model cannot be split at all", so a perfectly shardable model gets placed
    across cards layer-by-layer and nothing says why.
    """


def _unreachable(error: BaseException) -> bool:
    """Whether ``error`` means "couldn't read it" rather than "it isn't there".

    ``LocalEntryNotFoundError`` is the important one and reads backwards: the Hub
    raises it when it could *not reach the network* and found nothing cached, so
    despite the name it is a connectivity failure, not a missing file.
    ``RepositoryNotFoundError`` and ``EntryNotFoundError``, by contrast, are the
    Hub successfully telling us there is nothing there.
    """
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(error, (RepositoryNotFoundError, RevisionNotFoundError)):
        return False
    if isinstance(error, (LocalEntryNotFoundError, OfflineModeIsEnabled)):
        return True
    if isinstance(error, HfHubHTTPError):
        # 5xx is the Hub failing; 4xx is the Hub answering.
        response = getattr(error, "response", None)
        return getattr(response, "status_code", 0) >= 500
    return isinstance(error, (OSError, TimeoutError))


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
        # load — so give the tree an ordinary interleaver carrying rules that can
        # handle it either way: they stay inert (one attribute check per handled
        # location) unless they find something actually split while instrumenting.
        # Explicit rather than setdefault so a caller-supplied interleaver wins.
        if "interleaver" not in kwargs:
            from ..intervention.interleaver import Interleaver
            from .tp import TPFragments

            kwargs["interleaver"] = Interleaver(fragments=TPFragments())

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
        """Size the weights from the Hub's parameter count.

        The size half of
        [`_remoteable_describe_checkpoint`][nnsight.modeling.huggingface.HuggingFaceModel._remoteable_describe_checkpoint],
        for a caller that wants only that. Not a separate implementation — the
        two must not be able to disagree about how big a model is, since one of
        them decides where it goes.
        """
        return cls._remoteable_describe_checkpoint(
            model_key, dtype, trust_remote_code=trust_remote_code
        ).size_bytes

    @staticmethod
    def _hub_parameter_count(repo_id: str, revision: Optional[str]) -> Optional[int]:
        """Total parameters the Hub reports for a repo, or None if it doesn't.

        Raises:
            CheckpointUnreachable: if the Hub could not be read. The caller has a
                fallback for a repo that publishes no count; it should not use it
                for a repo it simply failed to ask.
        """
        from huggingface_hub import HfApi

        try:
            info = HfApi().model_info(
                repo_id, revision=revision, timeout=HUB_TIMEOUT_SECONDS
            )
        except Exception as error:
            if _unreachable(error):
                raise CheckpointUnreachable(
                    f"could not read {repo_id!r} from the Hub: {error}"
                ) from error
            return None

        safetensors = getattr(info, "safetensors", None)
        return getattr(safetensors, "total", None)

    @classmethod
    def _remoteable_describe_checkpoint(
        cls, model_key: str, dtype: str, trust_remote_code: bool = False
    ) -> "CheckpointInfo":
        """Describe the repo from its metadata: one config read, one Hub record.

        The whole reason this is one call. Every field here comes from something
        already fetched — the revision is in the key, the config is memoized by
        `_config`, and the parameter count and the size are the same Hub record
        read once — where asking question by question meant fetching the config
        twice for a checkpoint nobody had seen before.

        The Hub indexes every safetensors checkpoint and reports how many
        parameters it holds, so the number the base class builds a whole meta
        model to count is one request away. Buffers are not in that count (they
        aren't checkpoint tensors); they're a rounding error next to the
        parameters for a transformer, and the caller pads.
        """
        from .mixins.remotable import CheckpointInfo, bytes_per_element

        data = json.loads(model_key)
        repo_id, revision = data.get("repo_id"), data.get("revision")

        parameters = cls._hub_parameter_count(repo_id, revision)
        if parameters is None:
            # No published count — an older .bin-only repo, a private mirror.
            # Fall back to building the architecture to count it, which is what
            # the base class does, and take the config on the way past.
            logger.debug(
                f"No parameter count published for {repo_id!r}; sizing it by "
                "building the architecture instead"
            )
            size_bytes = super()._remoteable_estimate_bytes(
                model_key, dtype, trust_remote_code=trust_remote_code
            )
        else:
            size_bytes = math.ceil(parameters * bytes_per_element(dtype))

        return CheckpointInfo(
            size_bytes=size_bytes,
            n_params=parameters,
            config=cls._config(model_key, trust_remote_code),
            revision=revision,
        )

    @classmethod
    def _config(cls, model_key: str, trust_remote_code: bool) -> Optional[Any]:
        """The repo's config, fetched at most once per (key, trust) pair.

        Memoized because several of the questions a server asks before placing a
        model are all answered from this one object — its size, how many ways it
        shards, what to show in a status — and each used to fetch it again. A
        config is immutable for a pinned revision, and for an unpinned one the
        answer is "whatever the branch said when this process first asked", which
        is already true of every other read on this path.

        Returns ``None`` when the repo has no readable config.

        Raises:
            CheckpointUnreachable: if the Hub could not be read at all.
        """
        cache_key = (model_key, trust_remote_code)
        if cache_key in _CONFIG_CACHE:
            return _CONFIG_CACHE[cache_key]

        from transformers import AutoConfig

        data = json.loads(model_key)
        try:
            config = AutoConfig.from_pretrained(
                data.get("repo_id"),
                revision=data.get("revision"),
                trust_remote_code=trust_remote_code,
            )
        except Exception as error:
            if _unreachable(error):
                raise CheckpointUnreachable(
                    f"could not read a config for {data.get('repo_id')!r}: {error}"
                ) from error
            logger.debug(f"Could not read a config for {data.get('repo_id')!r}")
            _CONFIG_CACHE[cache_key] = None
            return None

        _CONFIG_CACHE[cache_key] = config
        return config

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

"""Pipeline Parallelism utilities for NNsight vLLM integration.

Provides detection of PPMissingLayer modules and a mapping from module
paths to their owning PP rank.
"""

from __future__ import annotations

import os
from typing import Optional

import torch.nn as nn


# ---------------------------------------------------------------------------
# PP timeouts (one home for the previously-inline magic numbers).
#
# Only the readiness-gate deadline carries a real false-trip risk: a forward
# waits for its workers to be "ahead", and a worker blocked in a slow UPSTREAM
# cross-stage pull (huge hidden state over a degraded cross-node link) reaches
# its local part late. That one is env-overridable. The poll/backoff are
# internal cadence (no false-trip risk), so those stay fixed.
#
# The override is an env var, not ``CONFIG``/``config.yaml``: these run in the
# vLLM WORKER process, and env vars propagate to Ray workers cross-node whereas
# the yaml would have to exist on every node.
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    """Read a float override from the environment, falling back to ``default``
    on absence or a malformed value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Readiness gate: how long a forward waits for its scheduled mediators to be
# ahead before raising loudly (vs hanging). Override: NNSIGHT_PP_GATE_TIMEOUT.
PP_GATE_TIMEOUT_S = _env_float("NNSIGHT_PP_GATE_TIMEOUT", 30.0)

# Finalize worker join — a safety net; the drain barrier normally completes the
# in-flight pull first, so this is rarely the limiter.
PP_FINALIZE_JOIN_S = 5.0

# Internal mechanics (not user-facing):
PP_GATE_POLL_S = 1e-4          # readiness-gate spin granularity
PP_LISTENER_BACKOFF_S = 0.5    # listener retry after a transient error


def resolve_meta(meta_map: dict, path: str):
    """Look up per-module PP metadata, tolerant of the nnsight root prefix.

    The metadata map (``pp_module_meta``) is keyed by vLLM's raw
    ``named_modules()`` names (e.g. ``"transformer.h.8"``), but cross-stage
    accesses look it up by the **nnsight envoy path**, which carries a root
    prefix (e.g. ``"model.transformer.h.8"``). A plain ``dict.get`` therefore
    misses and the pull silently falls back to its ``float32`` default — which
    then corrupts the dtype of any pulled value written back into the (bf16)
    model. Strip leading path components until a key matches.

    Returns the metadata dict, or ``None`` if no prefix variant matches.
    """
    if path in meta_map:
        return meta_map[path]
    parts = path.split(".")
    for i in range(1, len(parts)):
        cand = ".".join(parts[i:])
        if cand in meta_map:
            return meta_map[cand]
    return None


def is_pp_missing(module: nn.Module) -> bool:
    """Check whether *module* is a vLLM ``PPMissingLayer`` stub.

    vLLM replaces layers that don't belong to a PP rank with
    ``PPMissingLayer`` (a subclass of ``nn.Identity``).  This function
    checks by class name so we don't need a hard import of the vLLM
    internal class (which may move between versions).
    """
    return type(module).__name__ == "PPMissingLayer"


# Container names that hold the repeated transformer layers.
_LAYER_CONTAINER_NAMES = {"layers", "h", "block", "blocks"}

# Modules that always live on the first PP rank.
_FIRST_RANK_MODULES = {"embed_tokens", "wte", "wpe"}

# Modules that always live on the last PP rank.
# ``logits_processor`` is included because some architectures (Qwen2/GPT2/OPT/
# Pythia/Bloom/Gemma2) construct it unconditionally on every rank rather than as
# a ``PPMissingLayer`` stub, yet it only fires on the last rank — so direct
# access on a non-last rank must short-circuit to a cross-stage pull, not block
# on a hook that never fires.
_LAST_RANK_MODULES = {"norm", "lm_head", "ln_f", "logits", "samples", "logits_processor"}


class PPModuleMap:
    """Maps module attribute paths to the PP rank that owns them.

    Built once at model load time from ``num_hidden_layers`` and
    ``pp_world_size``.  Uses ``vllm.distributed.utils.get_pp_indices``
    to compute layer boundaries.

    Parameters
    ----------
    num_hidden_layers : int
        Total number of transformer layers in the model.
    pp_world_size : int
        Number of pipeline-parallel stages.
    """

    # Trailing eproperty keys to strip before resolving a module path.
    _EPROPERTY_KEYS = ("output", "input", "inputs")

    def __init__(self, num_hidden_layers: int, pp_world_size: int):
        from vllm.distributed.utils import get_pp_indices

        self.num_hidden_layers = num_hidden_layers
        self.pp_world_size = pp_world_size

        # Build per-rank layer ranges: rank -> (start, end)
        self._rank_ranges: dict[int, tuple[int, int]] = {}
        for rank in range(pp_world_size):
            start, end = get_pp_indices(num_hidden_layers, rank, pp_world_size)
            self._rank_ranges[rank] = (start, end)

        # Owning rank per REAL module path, derived from the load-time meta
        # exchange (which module is non-``PPMissingLayer`` on which stage).
        # Populated by ``set_derived_owners`` after the allgather; empty until
        # then, in which case ``get_owning_rank`` uses only the legacy
        # layer-range + name logic (so unit tests / PP-disabled are unchanged).
        self._derived_owners: dict[str, int] = {}

    def set_derived_owners(self, owners: dict) -> None:
        """Install the per-module owning-rank map derived from the meta
        exchange (``GPUModelRunner._exchange_pp_module_meta``).

        Keys are vLLM ``named_modules()`` names (e.g. ``"model.layers.5"``,
        ``"model.word_embeddings"``); values are PP stage indices. Modules real
        on more than one rank (containers, build-on-every-rank modules) are
        omitted by the exchange — those stay on the name-table fallback.
        """
        self._derived_owners = dict(owners)

    def _strip_eproperty(self, parts: list) -> list:
        """Drop a trailing eproperty key (``output``/``input``/``inputs``)."""
        if parts and parts[-1] in self._EPROPERTY_KEYS:
            return parts[:-1]
        return parts

    def _derived_owner(self, parts: list) -> Optional[int]:
        """Resolve ownership from the derived map for the longest owned ancestor
        of ``parts``, tolerant of the nnsight root prefix.

        A submodule (``...layers.5.mlp``) inherits the stage of its nearest
        owned ancestor (``...layers.5``); the prefix walk matches the vLLM raw
        key (``model.layers.5``) against the prefixed nnsight path
        (``model.model.layers.5``).
        """
        if not self._derived_owners:
            return None
        walk = list(parts)
        while walk:
            # Prefix-tolerant exact match (strip leading nnsight root parts).
            for i in range(len(walk)):
                cand = ".".join(walk[i:])
                if cand in self._derived_owners:
                    return self._derived_owners[cand]
            walk = walk[:-1]
        return None

    def get_owning_rank(self, module_path: str) -> Optional[int]:
        """Return the PP rank that owns *module_path*, or ``None`` if unknown.

        Resolution order:

        1. **Derived ownership** from the meta exchange — name-agnostic, covers
           every module real on exactly one stage (embeddings, norms, heads,
           layers, by their actual location regardless of naming convention).
        2. **Layer range** — the contiguous ``[start, end)`` slice (safety net
           and the path before any exchange).
        3. **First/last-rank name table** — only the non-derivable cases:
           modules built on every rank but firing only on the last
           (``logits``/``samples``/``logits_processor``), which are not distinct
           ``nn.Module``s in ``named_modules()``.

        Parameters
        ----------
        module_path : str
            Dot-separated attribute path, e.g. ``"model.layers.5"`` or
            ``"model.lm_head"``.
        """
        parts = self._strip_eproperty(module_path.split("."))

        owner = self._derived_owner(parts)
        if owner is not None:
            return owner

        # Check for layer container (e.g. model.layers.5.attn -> layer index 5)
        for i, part in enumerate(parts):
            if part in _LAYER_CONTAINER_NAMES and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except (ValueError, IndexError):
                    continue
                for rank, (start, end) in self._rank_ranges.items():
                    if start <= layer_idx < end:
                        return rank
                return None

        # Check for first/last rank modules
        for part in parts:
            if part in _FIRST_RANK_MODULES:
                return 0
            if part in _LAST_RANK_MODULES:
                return self.pp_world_size - 1

        return None

    def is_local(self, module_path: str, local_rank: int) -> bool:
        """Return whether *module_path* is owned by *local_rank*."""
        owner = self.get_owning_rank(module_path)
        if owner is None:
            # Unknown module -- assume local (safe default)
            return True
        return owner == local_rank

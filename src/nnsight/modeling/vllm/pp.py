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


class PPModuleMap:
    """Maps module attribute paths to the PP rank that owns them.

    Ownership is DERIVED, not guessed from naming conventions: the load-time
    meta exchange (``GPUModelRunner._exchange_pp_module_meta``) reports which
    modules are real (non-``PPMissingLayer``) on which stage, and the runner
    installs the result via :meth:`set_derived_owners` — adding explicit
    last-rank entries for the build-on-every-rank, fire-on-last modules
    (``logits``/``samples``/``logits_processor``), which the exchange cannot
    attribute. A module's stage is wherever it is real, regardless of what
    the architecture calls it.

    Parameters
    ----------
    pp_world_size : int
        Number of pipeline-parallel stages.
    """

    # Trailing eproperty keys to strip before resolving a module path.
    _EPROPERTY_KEYS = ("output", "input", "inputs")

    def __init__(self, pp_world_size: int):
        self.pp_world_size = pp_world_size

        # Owning rank per REAL module path, from the load-time meta exchange.
        # Empty until ``set_derived_owners`` runs (no trace executes before
        # model load completes), in which case every path resolves to ``None``
        # — the safe treated-as-local default.
        self._derived_owners: dict[str, int] = {}

    def set_derived_owners(self, owners: dict) -> None:
        """Install the per-module owning-rank map derived from the meta
        exchange.

        Keys are vLLM ``named_modules()`` names (e.g. ``"model.layers.5"``,
        ``"model.word_embeddings"``) plus the runner's structural last-rank
        claims; values are PP stage indices.
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

        ``None`` is treated as local by :meth:`is_local` (safe default); a
        genuine cross-stage consume of an unresolvable path raises a
        descriptive error at pull time (``pp_envoy._pp_lazy_access``).

        Parameters
        ----------
        module_path : str
            Dot-separated attribute path, e.g. ``"model.layers.5"`` or
            ``"model.lm_head"``.
        """
        return self._derived_owner(self._strip_eproperty(module_path.split(".")))

    def is_local(self, module_path: str, local_rank: int) -> bool:
        """Return whether *module_path* is owned by *local_rank*."""
        owner = self.get_owning_rank(module_path)
        if owner is None:
            # Unknown module -- assume local (safe default)
            return True
        return owner == local_rank

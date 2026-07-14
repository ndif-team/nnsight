"""Pipeline Parallelism utilities for NNsight vLLM integration.

Provides detection of PPMissingLayer modules and a mapping from module
paths to their owning PP rank.
"""

from __future__ import annotations

import os
from datetime import timedelta
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

# Pull-group gloo timeout. The listener blocks on an idle ``dist.recv`` for as
# long as a server sits between requests — that is BY DESIGN, not a hang. But
# ``dist.new_group`` otherwise inherits torch's 30-min default PG timeout, and
# gloo closes the whole peer pair when that expires on the idle recv, which
# permanently breaks every later cross-stage pull (a serve idle >30 min then
# 500s its first intervention). Pull safety is governed by the app-level
# park + error-reply protocol, not by this timeout, so set it effectively
# infinite. (Per-op recv timeouts are deliberately avoided too — see
# ``pp_listener`` module docstring on pair closure.)
PP_PULL_GROUP_TIMEOUT = timedelta(days=365)


def resolve_meta(meta_map: dict, path: str, root: str = "model"):
    """Look up per-module PP metadata by envoy path.

    The metadata map (``pp_module_meta``) is keyed by vLLM's raw
    ``named_modules()`` names (e.g. ``"transformer.h.8"``); an **nnsight envoy
    path** is, by Envoy construction, exactly the root envoy's name plus that
    raw name (``"model.transformer.h.8"``). So the lookup is: exact match
    (raw-name callers), else strip the single root component and match. A
    miss must not be papered over by stripping arbitrarily many components —
    a wrong-entry hit would silently stamp the wrong dtype.

    Returns the metadata dict, or ``None``.
    """
    if path in meta_map:
        return meta_map[path]
    prefix = root + "."
    if path.startswith(prefix):
        return meta_map.get(path[len(prefix):])
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
    root_path : str
        The root envoy's path (``Envoy``'s default root name). Every envoy
        path is, by construction, this single component followed by the
        module's raw ``named_modules()`` name — so lookups strip exactly one
        known component instead of guessing.
    """

    # Trailing eproperty keys to strip before resolving a module path.
    _EPROPERTY_KEYS = ("output", "input", "inputs")

    def __init__(self, pp_world_size: int, root_path: str = "model"):
        self.pp_world_size = pp_world_size
        self.root_path = root_path

        # Owning rank per REAL module path, from the load-time meta exchange.
        # Empty until ``set_derived_owners`` runs (no trace executes before
        # model load completes), in which case every path resolves to ``None``
        # — the safe treated-as-local default.
        self._derived_owners: dict[str, int] = {}
        # Ownership is constant for the model's life; memoize per lookup path
        # (``get_owning_rank`` runs on every cross-stage access).
        self._owner_memo: dict[str, Optional[int]] = {}

    def set_derived_owners(self, owners: dict) -> None:
        """Install the per-module owning-rank map derived from the meta
        exchange.

        Keys are vLLM ``named_modules()`` names (e.g. ``"model.layers.5"``,
        ``"model.word_embeddings"``) plus the runner's structural last-rank
        claims; values are PP stage indices.
        """
        self._derived_owners = dict(owners)
        self._owner_memo = {}

    def _strip_eproperty(self, parts: list) -> list:
        """Drop a trailing eproperty key (``output``/``input``/``inputs``)."""
        if parts and parts[-1] in self._EPROPERTY_KEYS:
            return parts[:-1]
        return parts

    def _derived_owner(self, parts: list) -> Optional[int]:
        """Resolve ownership for the longest owned ancestor of ``parts``.

        A submodule (``…layers.5.mlp``) inherits the stage of its nearest
        owned ancestor (``…layers.5``). The root-stripped form (envoy path →
        raw ``named_modules`` name) is tried first — the common case; the
        as-given form second (raw-name callers, e.g. error paths and tests).
        """
        if not self._derived_owners:
            return None
        stripped = parts[1:] if parts and parts[0] == self.root_path else None
        for candidate in (stripped, parts):
            if not candidate:
                continue
            walk = list(candidate)
            while walk:
                owner = self._derived_owners.get(".".join(walk))
                if owner is not None:
                    return owner
                walk.pop()
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
        if module_path not in self._owner_memo:
            self._owner_memo[module_path] = self._derived_owner(
                self._strip_eproperty(module_path.split("."))
            )
        return self._owner_memo[module_path]

    def is_local(self, module_path: str, local_rank: int) -> bool:
        """Return whether *module_path* is owned by *local_rank*."""
        owner = self.get_owning_rank(module_path)
        if owner is None:
            # Unknown module -- assume local (safe default)
            return True
        return owner == local_rank

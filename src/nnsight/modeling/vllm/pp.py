"""Pipeline-parallel ownership for the vLLM integration.

Under PP each rank holds only its stage's layers; vLLM replaces the rest with
``PPMissingLayer`` stubs. A trace, though, is written against the whole model,
and every rank runs every block — so each rank must know, per module path,
whether the value lives here or on another stage. This module provides that
resolution: :func:`is_pp_missing` detects the stubs, and :class:`PPModuleMap`
maps a path to its owning stage, derived from the load-time meta exchange
(each rank allgathers which modules are real for it) rather than guessed from
layer names.

Timeout policy lives here too. Overrides ride env vars, not ``CONFIG``: this
code runs in vLLM worker processes, and env vars propagate to cross-node Ray
workers whereas a yaml would have to exist on every node.
"""

from __future__ import annotations

import os
from typing import Optional

import torch.nn as nn


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


# How long a rank's serve point waits on an in-flight cross-stage pull before
# raising loudly (vs hanging). The one deadline with false-trip risk: a pull of
# a huge hidden state over a degraded cross-node link is slow but legitimate.
# Override: NNSIGHT_PP_PULL_TIMEOUT.
PP_PULL_TIMEOUT_S = _env_float("NNSIGHT_PP_PULL_TIMEOUT", 30.0)

# Listener retry after a transient error (internal cadence, no false-trip risk).
PP_LISTENER_BACKOFF_S = 0.5


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
    ``PPMissingLayer`` (a subclass of ``nn.Identity``). Checked by class name
    so we don't need a hard import of the vLLM internal class (which may move
    between versions).
    """
    return type(module).__name__ == "PPMissingLayer"


class PPModuleMap:
    """Maps module attribute paths to the PP rank that owns them.

    Ownership is DERIVED, not guessed from naming conventions: the load-time
    meta exchange (``NNsightGPUModelRunner``) reports which modules are real
    (non-``PPMissingLayer``) on which stage, and the runner installs the result
    via :meth:`set_derived_owners` — adding explicit last-rank entries for the
    build-on-every-rank, fire-on-last modules (``logits``/``samples``/
    ``logits_processor``), which the exchange cannot attribute. A module's
    stage is wherever it is real, regardless of what the architecture calls it.

    Args:
        pp_world_size: Number of pipeline-parallel stages.
        root_path: The root envoy's path. Every envoy path is, by
            construction, this single component followed by the module's raw
            ``named_modules()`` name — so lookups strip exactly one known
            component instead of guessing.
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
        """Install the per-module owning-rank map derived from the meta exchange.

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
        descriptive error at pull time.
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
            # Unknown module — assume local (safe default).
            return True
        return owner == local_rank

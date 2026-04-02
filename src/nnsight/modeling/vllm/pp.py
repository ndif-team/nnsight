"""Pipeline Parallelism utilities for NNsight vLLM integration.

Provides detection of PPMissingLayer modules and a mapping from module
paths to their owning PP rank.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn


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
_LAST_RANK_MODULES = {"norm", "lm_head", "ln_f", "logits", "samples"}


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

    def __init__(self, num_hidden_layers: int, pp_world_size: int):
        from vllm.distributed.utils import get_pp_indices

        self.num_hidden_layers = num_hidden_layers
        self.pp_world_size = pp_world_size

        # Build per-rank layer ranges: rank -> (start, end)
        self._rank_ranges: dict[int, tuple[int, int]] = {}
        for rank in range(pp_world_size):
            start, end = get_pp_indices(num_hidden_layers, rank, pp_world_size)
            self._rank_ranges[rank] = (start, end)

    def get_owning_rank(self, module_path: str) -> Optional[int]:
        """Return the PP rank that owns *module_path*, or ``None`` if unknown.

        Parameters
        ----------
        module_path : str
            Dot-separated attribute path, e.g. ``"model.layers.5"`` or
            ``"model.lm_head"``.
        """
        parts = module_path.split(".")

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

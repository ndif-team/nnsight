"""Batcher for HuggingFace continuous batching with paged attention.

HF CB packs all requests' tokens on dim 1 as ``[1, total_tokens, hidden_dim]``
(batch=1, tokens packed on sequence dimension).  This differs from the base
``Batcher`` which slices on dim 0, and from vLLM's ``VLLMBatcher`` which also
slices on dim 0 (vLLM uses 2D ``[total_tokens, hidden_dim]``).

The ``batch_dim`` parameter controls which dimension to narrow/swap on.
"""

from functools import partial
from typing import Any, List, Optional

import torch

from ...intervention.batching import Batcher, apply


class HFBatcher(Batcher):
    """Batcher for HF continuous batching.

    Slices on ``batch_dim`` (default 1) instead of dim 0, since HF CB
    packs all requests into ``[1, total_tokens, hidden_dim]``.
    """

    def __init__(self, *args, batch_dim: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_dim = batch_dim

    def _narrow(self, batch_group: Optional[List[int]], acts: torch.Tensor) -> torch.Tensor:
        batch_start, batch_size = batch_group

        if acts.dim() > self.batch_dim and acts.shape[self.batch_dim] == self.total_batch_size:
            return acts.narrow(self.batch_dim, batch_start, batch_size)

        return acts

    def _swap(self, batch_group: Optional[List[int]], current_value: torch.Tensor, swap_value: torch.Tensor) -> torch.Tensor:
        batch_start, batch_size = batch_group

        if current_value.dim() > self.batch_dim and current_value.shape[self.batch_dim] == self.total_batch_size:

            needs_concat = (
                current_value.requires_grad and current_value.is_leaf
            ) or current_value._base is not None

            if needs_concat:
                pre = current_value.narrow(self.batch_dim, 0, batch_start)
                post_start = batch_start + batch_size
                post_size = current_value.shape[self.batch_dim] - post_start
                post = current_value.narrow(self.batch_dim, post_start, post_size) if post_size > 0 else None

                parts = [pre, swap_value]
                if post is not None:
                    parts.append(post)
                return torch.cat(parts, dim=self.batch_dim)
            else:
                # In-place assignment on the batch_dim slice
                idx = [slice(None)] * self.batch_dim + [slice(batch_start, batch_start + batch_size)]
                current_value[tuple(idx)] = swap_value

        return current_value

    @property
    def total_batch_size(self):
        """Total number of packed tokens across all requests."""
        if self.last_batch_group is None:
            return 0
        return sum(self.last_batch_group)

"""
Example utility module for testing serialization of imported code.

This module provides realistic utilities that a researcher might use
when analyzing model internals with nnsight:

1. normalize() - A simple function for normalizing activations
2. RunningStats - A class for accumulating running mean/variance
3. ProjectionModule - A torch.nn.Module subclass for projecting activations

These are used by test_serialization_imports.py to verify that code
imported from external modules serializes correctly for remote='local'
and remote=True execution.
"""

import torch
import torch.nn as nn


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a tensor to unit norm along the specified dimension.

    This is a common operation when analyzing activation directions.

    Args:
        x: Input tensor
        dim: Dimension to normalize along (default: -1)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with unit norm along dim
    """
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def top_k_indices(x: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Get indices of top-k values along the last dimension.

    Useful for finding which features/tokens have highest activation.

    Args:
        x: Input tensor
        k: Number of top indices to return

    Returns:
        Indices of top-k values
    """
    return x.topk(k, dim=-1).indices


class RunningStats:
    """
    Accumulates running mean and variance statistics.

    Useful for computing statistics over many batches of activations
    without storing all the data.

    Example:
        stats = RunningStats()
        with model.trace(inputs):
            hidden = model.layer.output[0]
            stats.add(hidden)
        print(stats.mean(), stats.std())
    """

    def __init__(self):
        self.count = 0
        self._mean = None
        self._m2 = None  # Running sum of squared deviations

    def add(self, x: torch.Tensor) -> None:
        """
        Add a batch of observations.

        Uses Welford's online algorithm for numerical stability.

        Args:
            x: Tensor of shape [batch, ...]. Batch dimension is aggregated.
        """
        # Flatten batch dimension
        x = x.reshape(-1, *x.shape[2:]) if x.dim() > 2 else x.reshape(-1, x.shape[-1])
        batch_count = x.shape[0]

        if batch_count == 0:
            return

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        if self._mean is None:
            # First batch - initialize
            self.count = batch_count
            self._mean = batch_mean.clone()
            self._m2 = batch_var * batch_count
        else:
            # Update running statistics (Chan's parallel algorithm)
            new_count = self.count + batch_count
            delta = batch_mean - self._mean

            self._mean = self._mean + delta * (batch_count / new_count)
            self._m2 = self._m2 + batch_var * batch_count + \
                       delta.pow(2) * (self.count * batch_count / new_count)
            self.count = new_count

    def mean(self) -> torch.Tensor:
        """Return the running mean."""
        return self._mean

    def variance(self) -> torch.Tensor:
        """Return the running variance."""
        if self.count < 2:
            return torch.zeros_like(self._m2) if self._m2 is not None else None
        return self._m2 / self.count

    def std(self) -> torch.Tensor:
        """Return the running standard deviation."""
        var = self.variance()
        return var.sqrt() if var is not None else None

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self._mean = None
        self._m2 = None


class ProjectionModule(nn.Module):
    """
    A simple projection module for dimensionality reduction.

    Useful for projecting high-dimensional activations to a lower
    dimensional space for analysis or visualization.

    Example:
        proj = ProjectionModule(768, 64)
        with model.trace(inputs):
            hidden = model.layer.output[0]
            reduced = proj(hidden).save()
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to lower dimensional space."""
        return self.projection(x)

    def project_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize to unit vectors."""
        projected = self.projection(x)
        return normalize(projected)

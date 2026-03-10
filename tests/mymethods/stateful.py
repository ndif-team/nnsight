"""Test module with mutable state for verifying serialization isolation.

This module is used to test that when code is serialized and executed via
remote='local', it gets its own copy of the module state rather than
sharing state with the original.
"""
import torch

# Mutable module-level state
_call_count = 0
_last_value = None


def increment_and_get():
    """Increment the call count and return the new value.

    This function modifies module-level state, allowing us to verify
    that serialized code runs in an isolated environment.
    """
    global _call_count
    _call_count += 1
    return _call_count


def get_call_count():
    """Get the current call count without incrementing."""
    return _call_count


def reset_count():
    """Reset the call count to zero."""
    global _call_count
    _call_count = 0


def store_value(x):
    """Store a value in module-level state."""
    global _last_value
    _last_value = x
    return x


def get_stored_value():
    """Get the last stored value."""
    return _last_value


def normalize(x, dim=-1, eps=1e-8):
    """Normalize a tensor to unit norm.

    This is a simple utility function to test that user-defined
    functions work correctly with the whitelist-based serialization.
    """
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class RunningStats:
    """Accumulate running mean over batches.

    This class tests that user-defined classes are correctly
    serialized when they're from non-whitelisted modules.
    """

    def __init__(self):
        self.count = 0
        self._mean = None

    def add(self, x):
        """Add a batch to the running mean calculation."""
        batch_mean = x.mean()
        if self._mean is None:
            self._mean = batch_mean
        else:
            self._mean = (self._mean * self.count + batch_mean) / (self.count + 1)
        self.count += 1
        return self

    def mean(self):
        """Get the current running mean."""
        return self._mean if self._mean is not None else torch.tensor(0.0)

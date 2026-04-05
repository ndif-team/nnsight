"""LazyRemoteTensor — proxy for PPMissing module outputs.

Returned by the Envoy when accessing .output on a module that lives on
a different PP rank. Most operations are no-ops (writes, saves). Only
real tensor operations (arithmetic, torch functions) trigger
materialization via RPC pull from the source rank's listener.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch.utils._pytree import tree_map


class LazyRemoteTensor:
    """Proxy that materializes into a real tensor on first read operation."""

    def __init__(
        self,
        source_rank: int,
        provider_string: str,
        dtype: torch.dtype,
    ):
        self._meta = {
            "source_rank": source_rank,
            "provider_string": provider_string,
            "dtype": dtype,
        }
        self._real: torch.Tensor | None = None
        self._pull_fn = None  # set externally by whoever creates the lazy tensor

    def __getstate__(self):
        """Make picklable by excluding the unpicklable _pull_fn."""
        state = self.__dict__.copy()
        state["_pull_fn"] = None
        return state

    def _materialize(self) -> torch.Tensor:
        """Pull real tensor from source rank's listener.

        Blocks until the tensor is available.
        """
        if self._real is None:
            if self._pull_fn is None:
                raise RuntimeError(
                    f"Cannot materialize LazyRemoteTensor for "
                    f"{self._meta['provider_string']}: no pull function set."
                )
            self._real = self._pull_fn(
                self._meta["source_rank"],
                self._meta["provider_string"],
            )
        return self._real

    # --- torch interop: materialize on any real operation ---

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        args = tree_map(
            lambda x: x._materialize() if isinstance(x, LazyRemoteTensor) else x,
            args,
        )
        kwargs = tree_map(
            lambda x: x._materialize() if isinstance(x, LazyRemoteTensor) else x,
            kwargs,
        )
        # vLLM tensors are inference-mode; in-place ops on them
        # require inference_mode context.
        with torch.inference_mode():
            return func(*args, **kwargs)

    # --- arithmetic: materialize and delegate to real tensor ---
    # __torch_function__ only fires for explicit torch.* calls (e.g. torch.sum).
    # Python operators like + need dunder methods on the class itself.

    def __add__(self, other):
        return self._materialize() + other

    def __radd__(self, other):
        return other + self._materialize()

    def __sub__(self, other):
        return self._materialize() - other

    def __rsub__(self, other):
        return other - self._materialize()

    def __mul__(self, other):
        return self._materialize() * other

    def __rmul__(self, other):
        return other * self._materialize()

    def __truediv__(self, other):
        return self._materialize() / other

    def __rtruediv__(self, other):
        return other / self._materialize()

    def __neg__(self):
        return -self._materialize()

    def __matmul__(self, other):
        return self._materialize() @ other

    def __rmatmul__(self, other):
        return other @ self._materialize()

    # --- no-op absorbers ---

    def __setitem__(self, key: Any, value: Any) -> None:
        pass  # absorb writes without materialization

    def __getitem__(self, key: Any) -> "LazyRemoteTensor":
        # Return a new LazyRemoteTensor that applies the index after materialization.
        child = LazyRemoteTensor(
            source_rank=self._meta["source_rank"],
            provider_string=self._meta["provider_string"],
            dtype=self._meta["dtype"],
        )
        parent = self
        index = key

        def _deferred_pull(source_rank, provider_string):
            return parent._materialize()[index]

        child._pull_fn = _deferred_pull
        return child

    def save(self) -> "LazyRemoteTensor":
        return self  # no-op on non-owning rank

    # --- method-style access: materialize and delegate ---
    # .mean(), .sum(), .view(), .float(), etc. go through __getattr__

    _OWN_ATTRS = frozenset({
        "_meta", "_real", "_pull_fn",
        "shape", "dtype", "device",
        "save", "_materialize",
    })

    def __getattr__(self, name: str):
        if name.startswith("_") or name in LazyRemoteTensor._OWN_ATTRS:
            raise AttributeError(name)
        return getattr(self._materialize(), name)

    # --- metadata (no materialization) ---

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._materialize().shape

    @property
    def dtype(self) -> torch.dtype:
        if self._real is not None:
            return self._real.dtype
        return self._meta["dtype"]

    @property
    def device(self) -> torch.device:
        return self._materialize().device

    def __repr__(self) -> str:
        status = "materialized" if self._real is not None else "lazy"
        return (
            f"LazyRemoteTensor({status}, "
            f"src=rank{self._meta['source_rank']}, "
            f"key={self._meta['provider_string']!r})"
        )

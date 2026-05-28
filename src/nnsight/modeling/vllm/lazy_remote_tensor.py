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


class _NotOnThisRankType:
    """Placeholder for a saved value (or container slot) owned by a different
    PP rank.

    When a saved value contains a :class:`LazyRemoteTensor` — e.g. a list of
    per-step activations where some elements live on another stage — the
    non-owning rank cannot ship a real tensor for that slot. Instead of
    shipping an un-materializable lazy (whose ``_pull_fn`` is dropped on
    pickle), it ships this sentinel. The engine then merges the per-rank
    saves position-wise (:func:`merge_saved`), filling each slot from the
    rank that actually owns it. Singleton so identity checks and pickling
    round-trip cleanly across the worker→engine boundary.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self):
        return (_NotOnThisRankType, ())

    def __repr__(self):
        return "NOT_ON_THIS_RANK"


NOT_ON_THIS_RANK = _NotOnThisRankType()


def strip_lazy(value):
    """Replace every :class:`LazyRemoteTensor` in ``value`` with the
    :data:`NOT_ON_THIS_RANK` sentinel, recursing into lists/tuples/dicts.

    Returns ``(stripped, has_real, has_lazy)``:
      * ``stripped`` — same structure with lazies swapped for the sentinel,
      * ``has_real`` — whether any non-lazy leaf is present (this rank owns
        at least one slot),
      * ``has_lazy`` — whether any lazy was found (some slot is owned
        elsewhere).

    Callers ship ``stripped`` only when the value isn't *purely* owned by
    another rank (``has_lazy and not has_real``); in that case the owning
    rank ships the real data and this rank contributes nothing.
    """
    if isinstance(value, LazyRemoteTensor):
        return NOT_ON_THIS_RANK, False, True
    if isinstance(value, (list, tuple)):
        stripped, has_real, has_lazy = [], False, False
        for item in value:
            s, r, l = strip_lazy(item)
            stripped.append(s)
            has_real |= r
            has_lazy |= l
        return type(value)(stripped), has_real, has_lazy
    if isinstance(value, dict):
        stripped, has_real, has_lazy = {}, False, False
        for k, item in value.items():
            s, r, l = strip_lazy(item)
            stripped[k] = s
            has_real |= r
            has_lazy |= l
        return stripped, has_real, has_lazy
    # Leaf that isn't lazy (real tensor, scalar, str, …): owned by this rank.
    return value, True, False


def merge_saved(a, b):
    """Position-wise merge of two same-shaped saved values from different PP
    ranks, preferring the non-:data:`NOT_ON_THIS_RANK` leaf at each slot.

    Used by the engine to assemble one complete result from each rank's
    partial contribution. If both sides are real leaves (or the structures
    don't line up), ``b`` wins — preserving the previous "later-rank-wins"
    merge semantics for scalars and degrading safely on mismatch.
    """
    if a is NOT_ON_THIS_RANK:
        return b
    if b is NOT_ON_THIS_RANK:
        return a
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [merge_saved(x, y) for x, y in zip(a, b)]
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return tuple(merge_saved(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return {k: merge_saved(a[k], b[k]) for k in a}
    return b

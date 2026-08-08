"""LazyRemoteTensor — proxy for values owned by another PP stage.

Served by the PP interleaver's intercept when a block reads ``.output`` /
``.input`` on a module that lives on a different PP rank. Handing back a proxy
instead of parking keeps the worker running to its LOCAL accesses (run-ahead: a
cross-stage read followed by a local read in the same step must not suspend the
worker before it registers the local access), and makes an unconsumed remote
read cost nothing — no pull, no traffic.

Only real consumption (arithmetic, torch functions, iteration, attribute
access) materializes the proxy. Materialization parks the worker on a synthetic
pull location; the PP interleaver issues the cross-stage pull the moment that
park happens (so the transfer overlaps the rest of the forward — see
``pp_listener``) and resumes the worker with the real tensor at its serve
point. Writes and saves are absorbed: the owning rank executes the same block
line locally, so the write happens there, and a saved proxy ships as the
:data:`NOT_ON_THIS_RANK` sentinel for the engine-side merge to fill from the
owning rank.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch.utils._pytree import tree_map

from ...intervention.interleaver import Mediator

# Prefix marking a synthetic pull-park location. The full encoded form is
# ``{PULL_LOCATION_PREFIX}{source_rank}::{req_id}::{provider}`` — parsed by
# ``decode_pull_location``; the provider (which itself contains dots) is the
# verbatim remainder. The PP interleaver recognizes the prefix in a worker's
# pending event and in its own intercept.
PULL_LOCATION_PREFIX = "__pp_pull__::"


def encode_pull_location(source_rank: int, req_id: Optional[str], provider: str) -> str:
    """Encode a pull's identity into the location string its park carries."""
    return f"{PULL_LOCATION_PREFIX}{source_rank}::{req_id if req_id is not None else ''}::{provider}"


def decode_pull_location(location: str) -> tuple[int, Optional[str], str]:
    """Inverse of :func:`encode_pull_location` → ``(source_rank, req_id, provider)``.

    Tolerates the ``.i{n}`` occurrence tag :meth:`Mediator.event` appends to
    every parked location — the provider inside the encoding is already
    occurrence-tagged at creation time, so a trailing tag is protocol noise and
    is stripped.
    """
    body = location[len(PULL_LOCATION_PREFIX):]
    rank_str, req_id_str, provider = body.split("::", 2)
    # Strip the park's own ".i{n}" suffix if present (the provider always ends
    # with its real occurrence tag, e.g. "...output.i0", so a doubled tag is
    # exactly one extra ".i{n}" component).
    parts = provider.rsplit(".", 2)
    if (
        len(parts) == 3
        and parts[2].startswith("i")
        and parts[2][1:].isdigit()
        and parts[1].startswith("i")
        and parts[1][1:].isdigit()
    ):
        provider = f"{parts[0]}.{parts[1]}"
    return int(rank_str), req_id_str or None, provider


class LazyRemoteTensor:
    """Proxy that materializes into a real tensor on first read operation."""

    def __init__(
        self,
        source_rank: int,
        provider_string: str,
        dtype: Optional[torch.dtype],
        req_id: Optional[str] = None,
    ):
        self._meta = {
            "source_rank": source_rank,
            "provider_string": provider_string,
            "dtype": dtype,
            "req_id": req_id,
        }
        self._real: Any = None
        # A deferred child (see __getitem__) materializes its parent and
        # indexes into it rather than pulling itself.
        self._parent: Optional["LazyRemoteTensor"] = None
        self._index: Any = None

    def _materialize(self) -> Any:
        """Park for the real value on first consumption; cached afterwards.

        Runs on the worker greenlet: the park suspends this block only — the
        PP interleaver issues the pull immediately (transfer overlaps the rest
        of the forward) and resumes the worker with the value at its serve
        point. Outside a worker there is nothing to park on and nothing to
        answer with, so a proxy that leaks past its trace raises.
        """
        if self._real is None:
            if self._parent is not None:
                self._real = self._parent._materialize()[self._index]
            else:
                self._real = Mediator.value(
                    encode_pull_location(
                        self._meta["source_rank"],
                        self._meta["req_id"],
                        self._meta["provider_string"],
                    )
                )
        return self._real

    # --- torch interop: materialize on any real operation ---

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Materializing here would park the worker greenlet INSIDE torch's
        # dispatcher: every torch binding installs thread-local state on entry
        # (the pybind warning handler lives on this greenlet's C stack) that
        # stays installed across the switch, and the forward that then runs on
        # this thread segfaults in C++ the next time that state is consulted
        # (observed: SIGSEGV in c10::warn under ProcessGroupNCCL::send).
        # Parks from plain Python frames — operators, methods, properties —
        # carry no dispatcher state, so force the value through one of those
        # first. Refusing loudly here turns a silent worker-process death into
        # a catchable in-trace error.
        def check(x):
            if isinstance(x, LazyRemoteTensor):
                root = x
                while root._parent is not None:
                    root = root._parent
                # A resolved root means materializing is pure indexing into
                # the cached value; only an unresolved root would park. With
                # the thread-local-state swap active (prototype), such parks
                # are the case it exists to make safe, so the guard stands
                # down and lets the canary exercise them.
                from .pp_tls_swap import active as _tls_swap_active

                if root._real is None and not _tls_swap_active():
                    raise RuntimeError(
                        f"{func.__name__} received a cross-stage value that "
                        f"has not been materialized "
                        f"({x._meta['provider_string']!r}). Torch functions "
                        f"cannot force one under pipeline parallelism, and "
                        f"an operator with a plain tensor on the LEFT "
                        f"(``tensor + value``) routes through the same torch "
                        f"machinery. Read the value first with a method or "
                        f"with it leading the expression (``value.clone()``, "
                        f"``value + 0``), then use the result."
                    )
            return x

        tree_map(check, (args, kwargs))
        args = tree_map(
            lambda x: x._materialize() if isinstance(x, LazyRemoteTensor) else x,
            args,
        )
        kwargs = tree_map(
            lambda x: x._materialize() if isinstance(x, LazyRemoteTensor) else x,
            kwargs,
        )
        # vLLM tensors are inference-mode; in-place ops on them require the
        # inference_mode context.
        with torch.inference_mode():
            return func(*args, **kwargs)

    # --- arithmetic: materialize and delegate to the real tensor ---
    # __torch_function__ only fires for explicit torch.* calls (e.g. torch.sum);
    # Python operators need dunder methods on the class itself.

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

    # --- comparisons: materialize and delegate ---
    # Without these, ``==``/``!=`` fall back to identity (a plain bool instead
    # of an elementwise tensor) and the orderings raise — but only on the
    # non-owning rank, so user code branching on a comparison silently diverges
    # between ranks rather than erroring.

    def __eq__(self, other):
        return self._materialize() == other

    def __ne__(self, other):
        return self._materialize() != other

    def __lt__(self, other):
        return self._materialize() < other

    def __le__(self, other):
        return self._materialize() <= other

    def __gt__(self, other):
        return self._materialize() > other

    def __ge__(self, other):
        return self._materialize() >= other

    # Defining ``__eq__`` clears the default hash; restore identity hashing —
    # the proxy is tracked by identity (the save set stores ``id``s), never by
    # value.
    __hash__ = object.__hash__

    # --- no-op absorbers ---

    def __setitem__(self, key: Any, value: Any) -> None:
        pass  # absorb writes without materialization; the owning rank performs them

    def __getitem__(self, key: Any) -> "LazyRemoteTensor":
        # A deferred child: applies the index after (its parent's)
        # materialization, so indexing alone still pulls nothing.
        child = LazyRemoteTensor(
            source_rank=self._meta["source_rank"],
            provider_string=self._meta["provider_string"],
            dtype=self._meta["dtype"],
            req_id=self._meta["req_id"],
        )
        child._parent = self
        child._index = key
        return child

    def __iter__(self):
        # Iteration is a real "consume the whole value" operation, so it must
        # materialize — like arithmetic and ``__getattr__`` below. Without
        # this, Python falls back to the sequence protocol over
        # ``__getitem__``, which returns a fresh lazy for every index and never
        # raises ``IndexError``, so ``tuple(lazy)`` / ``for x in lazy`` /
        # unpacking spin forever on the non-owning rank while the owning rank
        # terminates — a divergence that hangs cross-stage replacement writes.
        return iter(self._materialize())

    def __len__(self) -> int:
        # Same rationale as ``__iter__``: ``len(lazy)`` must reflect the real
        # value, not silently differ between ranks.
        return len(self._materialize())

    # NOTE: ``.save()`` is deliberately NOT defined here. The mounted
    # ``object.save`` (or ``nnsight.save``) marks the proxy's id like any other
    # value, so the saved name ships from this rank as a NOT_ON_THIS_RANK
    # sentinel (see the save-collection strip) and the engine-side merge fills
    # it from the owning rank. Normal attribute lookup finds the mounted method
    # before ``__getattr__``, so no materialization happens either.

    # --- method-style access: materialize and delegate ---
    # .mean(), .sum(), .view(), .float(), etc. go through __getattr__.

    _OWN_ATTRS = frozenset({
        "_meta", "_real", "_parent", "_index",
        "shape", "dtype", "device", "_materialize",
    })

    def __getattr__(self, name: str):
        if name.startswith("_") or name in LazyRemoteTensor._OWN_ATTRS:
            raise AttributeError(name)
        real = self._materialize()
        # Multi-output modules (e.g. Qwen2 / Llama decoder blocks) produce a
        # ``(hidden, residual)`` tuple, not a tensor. Forwarding a tensor
        # method like ``.mean`` onto the materialized tuple gives a confusing
        # ``'tuple' object has no attribute 'mean'`` error; say what to do
        # instead — ``lazy[0]`` already returns a deferred child that pulls the
        # indexed element.
        if isinstance(real, tuple) and not hasattr(tuple, name):
            raise AttributeError(
                f"LazyRemoteTensor at "
                f"{self._meta['provider_string']!r} materialized to a "
                f"{len(real)}-tuple, not a tensor; cannot access tensor "
                f"attribute {name!r}. This module's output is a tuple — "
                f"index it first (e.g. ``lazy[0].{name}``) to operate on a "
                f"single tuple element."
            )
        return getattr(real, name)

    # --- metadata ---

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._materialize().shape

    @property
    def dtype(self) -> Optional[torch.dtype]:
        # The load-time meta exchange's hint, until the real value knows better.
        if self._real is not None and isinstance(self._real, torch.Tensor):
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
    """Placeholder for a saved value (or container slot) owned by another PP rank.

    When a saved value contains a :class:`LazyRemoteTensor` — e.g. a list of
    per-step activations where some elements live on another stage — the
    non-owning rank cannot ship a real tensor for that slot. It ships this
    sentinel instead, and the engine merges the per-rank saves position-wise,
    filling each slot from the rank that actually owns it. Singleton so
    identity checks and pickling round-trip cleanly across the worker→engine
    boundary.
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

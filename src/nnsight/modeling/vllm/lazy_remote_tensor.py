"""LazyRemoteTensor — proxy for PPMissing module outputs.

Returned by the Envoy when accessing .output on a module that lives on
a different PP rank. Most operations are no-ops (writes, saves). Only
real tensor operations (arithmetic, torch functions) trigger
materialization via RPC pull from the source rank's listener.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

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

    # --- comparisons: materialize and delegate ---
    # Without these, ``==``/``!=`` fall back to identity (a plain bool instead
    # of an elementwise tensor) and the orderings raise — but only on the
    # non-owning rank, so user code branching on a comparison silently
    # diverges between ranks rather than erroring.

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
    # the proxy is tracked by identity (e.g. ``Globals.saves`` stores ``id``s),
    # never by value.
    __hash__ = object.__hash__

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

    def __iter__(self):
        # Iteration is a real "consume the whole value" operation, so it must
        # materialize — like arithmetic and ``__getattr__`` below. Without this,
        # Python falls back to the sequence protocol over ``__getitem__``, which
        # returns a fresh lazy for every index and never raises ``IndexError``,
        # so ``tuple(lazy)`` / ``list(lazy)`` / ``for x in lazy`` / unpacking
        # spin forever on the non-owning rank (the owning rank iterates the real
        # value and terminates). That divergence hangs cross-stage replacement
        # writes such as ``layer.output = (out[0] + d,) + tuple(out[1:])``.
        return iter(self._materialize())

    def __len__(self) -> int:
        # Same rationale as ``__iter__``: ``len(lazy)`` must reflect the real
        # value, not silently differ between ranks (or fall through to a
        # confusing error on the non-owning rank).
        return len(self._materialize())

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
        real = self._materialize()
        # Multi-output modules (e.g. Qwen2 / Llama decoder blocks) produce a
        # ``(hidden, residual)`` tuple, not a tensor. Forwarding a tensor
        # method like ``.mean`` onto the materialized tuple gives a confusing
        # ``'tuple' object has no attribute 'mean'`` error. Detect the case
        # and tell the caller to index first — ``lazy[0]`` already returns a
        # deferred child lazy that pulls the indexed element.
        if isinstance(real, tuple) and not hasattr(tuple, name):
            raise AttributeError(
                f"LazyRemoteTensor at "
                f"{self._meta['provider_string']!r} materialized to a "
                f"{len(real)}-tuple, not a tensor; cannot access tensor "
                f"attribute {name!r}. This module's output is a tuple — "
                f"index it first (e.g. ``lazy[0].{name}``) to operate on "
                f"a single tuple element."
            )
        return getattr(real, name)

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


def _rebuild_sequence(original, items):
    """Rebuild a list/tuple of ``items`` with ``original``'s type.

    NamedTuple constructors take positional fields, not an iterable —
    ``type(original)(items)`` raises TypeError for them, killing the whole
    save-collection pass. Detect via ``_fields`` and splat.
    """
    cls = type(original)
    if isinstance(original, tuple) and hasattr(original, "_fields"):
        return cls(*items)
    return cls(items)


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
        return _rebuild_sequence(value, stripped), has_real, has_lazy
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


class PPRankDivergenceWarning(RuntimeWarning):
    """Two PP ranks produced *different* values for the same saved slot.

    Each PP rank executes the intervention body independently; a saved value
    not derived deterministically from model state — most commonly
    ``torch.randn`` & friends called INSIDE the trace — differs per rank.
    The merge keeps one rank's copy, which may not be the copy that was
    actually applied to the model on the owning stage. Generate randomness
    OUTSIDE the trace instead: pre-trace values are serialized with the
    intervention and arrive identical on every rank.
    """


def _divergence_detail(a, b) -> Optional[str]:
    """A short human-readable description of how ``a`` and ``b`` differ, or
    ``None`` if they are equivalent (or cannot be compared).

    Float tensors compare with a TIGHT tolerance (``rtol=1e-5, atol=1e-8``,
    ``equal_nan=True``): redundant cross-rank execution of the same math from
    the same inputs is deterministic up to low-order kernel noise, while the
    divergence worth flagging (desynced RNG, per-rank environment values) is
    orders of magnitude larger. Integer/bool tensors and scalars compare
    exactly. Incomparable values are treated as equivalent — a tripwire must
    not false-positive on exotic user types.
    """
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return f"type mismatch: {type(a).__name__} vs {type(b).__name__}"
        if a.shape != b.shape or a.dtype != b.dtype:
            return (
                f"tensor mismatch: shape {tuple(a.shape)}/{a.dtype} vs "
                f"{tuple(b.shape)}/{b.dtype}"
            )
        try:
            x, y = a.detach(), b.detach()
            if x.device != y.device:
                x, y = x.cpu(), y.cpu()
            if x.dtype.is_floating_point or x.dtype.is_complex:
                if not torch.allclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=True):
                    return f"max|Δ| = {(x - y).abs().max().item():.3g}"
            elif not torch.equal(x, y):
                return "integer/bool tensors differ"
        except Exception:
            return None
        return None
    if isinstance(a, (list, tuple, dict)) or isinstance(b, (list, tuple, dict)):
        # Only structurally-mismatched containers reach the leaf fallthrough
        # (matched ones recursed above) — report the shape of the mismatch
        # rather than deep-comparing (elements may be tensors).
        def _desc(v):
            return (
                f"{type(v).__name__} of len {len(v)}"
                if isinstance(v, (list, tuple, dict))
                else type(v).__name__
            )

        return f"structure mismatch: {_desc(a)} vs {_desc(b)}"
    try:
        if bool(a == b):
            return None
        ra, rb = repr(a)[:80], repr(b)[:80]
        return f"{ra} vs {rb}"
    except Exception:
        return None


def _warn_divergence(label, detail):
    warnings.warn(
        f"PP merge: ranks returned different values for saved slot "
        f"'{label or '<unnamed>'}' ({detail}). Each PP rank runs the "
        f"intervention body independently; the value kept is one rank's copy "
        f"and may not be the one applied to the model. If this comes from "
        f"randomness inside the trace (torch.randn etc.), generate it "
        f"OUTSIDE the trace — pre-trace values ship identically to every "
        f"rank. See docs/models/vllm.md (pipeline-parallel semantics).",
        PPRankDivergenceWarning,
        stacklevel=2,
    )


def _extend(label, part):
    return f"{label}{part}" if label else None


def merge_saved(a, b, label: Optional[str] = None):
    """Position-wise merge of two same-shaped saved values from different PP
    ranks, preferring the non-:data:`NOT_ON_THIS_RANK` leaf at each slot.

    Used by the engine to assemble one complete result from each rank's
    partial contribution. Lists/tuples merge element-wise; dicts merge by
    key union (so disjoint-keyed per-stage CacheDicts are combined, not
    dropped). If both sides are real leaves (or the structures don't line
    up), ``b`` wins — preserving the previous "later-rank-wins" merge
    semantics for scalars and degrading safely on mismatch.

    A real/real slot whose two copies genuinely DIFFER is the rank-divergence
    tripwire: it emits :class:`PPRankDivergenceWarning` (with ``label``
    naming the slot, e.g. ``"noise"`` or ``"outs[2]"``) instead of silently
    keeping an arbitrary copy. Identical redundant copies — the normal case
    for every value both ranks computed deterministically — merge silently.
    """
    if a is NOT_ON_THIS_RANK:
        return b
    if b is NOT_ON_THIS_RANK:
        return a
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            merge_saved(x, y, _extend(label, f"[{i}]"))
            for i, (x, y) in enumerate(zip(a, b))
        ]
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return _rebuild_sequence(
            a,
            [
                merge_saved(x, y, _extend(label, f"[{i}]"))
                for i, (x, y) in enumerate(zip(a, b))
            ],
        )
    if isinstance(a, dict) and isinstance(b, dict):
        # Union the key sets, merging overlapping slots position-wise
        # (sentinel-aware) and taking each disjoint key from whichever rank
        # owns it. A normal dict save carries identical keys on every rank
        # (non-owned slots are NOT_ON_THIS_RANK sentinels), so the union
        # reduces to the prior element-wise merge. But PP ``tracer.cache()``
        # hooks fire only on the modules that execute on each stage, so the
        # two ranks' caches have DISJOINT keys; the old equal-keys-only branch
        # fell through to ``return b`` and silently dropped a whole stage. The
        # union keeps every stage's contributions.
        #
        # Both sides arrive as plain ``dict``s here: ``strip_lazy`` rebuilds a
        # plain dict on the worker before collection, so a ``CacheDict`` is
        # already flattened by the time it reaches the merge (which is why
        # attribute access on a vLLM cache is unavailable at any PP level — a
        # pre-existing limitation, not introduced here). ``dict.__*`` is used
        # defensively so a dict subclass with overridden lookups can't corrupt
        # the copy.
        merged = {}
        for k in dict.__iter__(a):
            other = dict.__getitem__(b, k) if dict.__contains__(b, k) else NOT_ON_THIS_RANK
            merged[k] = merge_saved(
                dict.__getitem__(a, k), other, _extend(label, f"[{k!r}]")
            )
        for k in dict.__iter__(b):
            if not dict.__contains__(a, k):
                merged[k] = dict.__getitem__(b, k)
        return merged

    # Leaf fallthrough: both sides are real. Identical redundant copies are
    # the normal case (every rank computed the value deterministically from
    # the same inputs); a genuine difference means the ranks diverged —
    # in-trace randomness, a per-rank side effect, or structurally mismatched
    # save trees (e.g. one rank's worker appended fewer iteration steps).
    # ``b`` still wins (degrade the same way as before), but loudly.
    detail = _divergence_detail(a, b)
    if detail is not None:
        _warn_divergence(label, detail)
    return b

"""Assemble per-rank save payloads into one result (the PP merge).

Every PP stage runs the intervention body and ships a partial save tree from
its TP-rank-0: the slots it owns are real values, foreign slots are
:data:`~.lazy_remote_tensor.NOT_ON_THIS_RANK` sentinels (a saved
LazyRemoteTensor strips to one before shipping — see :func:`strip_lazy`). This
module is the single home for turning the per-rank ``collect_nnsight`` results
into one merged ``{engine_id: {"saves": ..., "error": ...}}``: the sync
engine, the async backend, and the serve handler all call
:func:`merge_collected` instead of taking the first non-``None`` payload
(which, under PP, silently drops every stage but one).

The organizing principle of :func:`merge_saved` is **stage ownership, encoded
by the sentinel**: "prefer the real leaf over the sentinel" *is* "trust the
owning rank". A non-owning rank that holds a real copy of a model-derived
value materialized the owner's (via the cross-stage pull), so two real copies
of anything model-derived are equal; only model-INDEPENDENT values (in-trace
RNG, per-rank environment values) can genuinely differ, and those trip
:class:`PPRankDivergenceWarning`.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Optional

import torch

from ...intervention.cache import Cache, CacheView, Entry
from .lazy_remote_tensor import NOT_ON_THIS_RANK, LazyRemoteTensor


def _rebuild_sequence(original, items):
    """Rebuild a list/tuple of ``items`` with ``original``'s type.

    NamedTuple constructors take positional fields, not an iterable —
    ``type(original)(items)`` raises for them. Detect via ``_fields`` and splat.
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
      * ``has_real`` — whether any non-lazy leaf is present (this rank owns at
        least one slot),
      * ``has_lazy`` — whether any lazy was found (some slot is owned
        elsewhere).

    Callers ship ``stripped`` only when the value isn't *purely* owned by
    another rank (``has_lazy and not has_real``); in that case the owning rank
    ships the real data and this rank contributes nothing.
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
        stripped_dict, has_real, has_lazy = {}, False, False
        for k, item in value.items():
            s, r, l = strip_lazy(item)
            stripped_dict[k] = s
            has_real |= r
            has_lazy |= l
        return stripped_dict, has_real, has_lazy
    return value, True, False


class PPRankDivergenceWarning(RuntimeWarning):
    """Two PP ranks produced *different* values for the same saved slot.

    Each PP rank executes the intervention body independently; a saved value
    not derived deterministically from model state — most commonly
    ``torch.randn`` & friends called INSIDE the trace — differs per rank. The
    merge keeps one rank's copy, which may not be the copy actually applied to
    the model on the owning stage. Generate randomness OUTSIDE the trace
    instead: pre-trace values are serialized with the intervention and arrive
    identical on every rank.
    """


def _divergence_detail(a, b) -> Optional[str]:
    """A short description of how ``a`` and ``b`` differ, or ``None`` if they
    are equivalent (or cannot be compared).

    Float tensors compare with a TIGHT tolerance (``rtol=1e-5, atol=1e-8``,
    ``equal_nan=True``): redundant cross-rank execution of the same math from
    the same inputs is deterministic up to low-order kernel noise, while the
    divergence worth flagging (desynced RNG, per-rank environment values) is
    orders of magnitude larger. Integer/bool tensors and scalars compare
    exactly. Incomparable values are treated as equivalent — a tripwire must
    not false-positive on exotic user types.
    """
    if a is b:
        return None
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
        # Only structurally-mismatched containers reach this leaf fallthrough
        # (matched ones recursed in merge_saved) — report the shape of the
        # mismatch rather than deep-comparing (elements may be tensors).
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
        return f"{repr(a)[:80]} vs {repr(b)[:80]}"
    except Exception:
        return None


def _warn_divergence(label, detail):
    warnings.warn(
        f"PP merge: ranks returned different values for saved slot "
        f"'{label or '<unnamed>'}' ({detail}). Each PP rank runs the "
        f"intervention body independently; the value kept is one rank's copy "
        f"and may not be the one applied to the model. If this comes from "
        f"randomness inside the trace (torch.randn etc.), generate it OUTSIDE "
        f"the trace — pre-trace values ship identically to every rank.",
        PPRankDivergenceWarning,
        stacklevel=2,
    )


def _extend(label, part):
    return f"{label}{part}" if label else None


def _has_real(v) -> bool:
    """Whether ``v`` contains any real (non-sentinel) leaf.

    A value owned by no rank at this position — a bare sentinel, or a
    container of nothing but sentinels — carries no data. Used to tell a
    rank's genuine contribution apart from run-ahead overshoot (a worker that
    looped one step past generation end appends an unconsumed lazy, which
    strips to a sentinel).
    """
    if v is NOT_ON_THIS_RANK:
        return False
    if isinstance(v, (list, tuple)):
        return any(_has_real(x) for x in v)
    if isinstance(v, dict):
        return any(_has_real(x) for x in dict.values(v))
    return True


def _real_beyond_common(a, b) -> int:
    """How many positions that only ONE of ``a``/``b`` reached carry real data.

    Separates the three ways two ranks' list lengths can differ: a non-owner's
    same-length all-sentinel list (0), run-ahead overshoot with a trailing
    sentinel (0), and a stalled/errored worker or stage-divergent control flow
    (real entries the other rank never produced — > 0). Only the last warns;
    the first two merge silently.
    """
    m = min(len(a), len(b))
    return sum(_has_real(x) for x in a[m:]) + sum(_has_real(x) for x in b[m:])


def _union_sequence(a, b, label):
    """Positional union of two sequences of possibly-unequal length: merge
    each shared position, take a position only one side reached as-is, then
    drop the trailing no-real overshoot tail. Returns a plain ``list``.
    """
    merged = []
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else NOT_ON_THIS_RANK
        y = b[i] if i < len(b) else NOT_ON_THIS_RANK
        merged.append(merge_saved(x, y, _extend(label, f"[{i}]")))
    while merged and not _has_real(merged[-1]):
        merged.pop()
    return merged


def merge_saved(a, b, label: Optional[str] = None):
    """Position-wise union of two saved values from different PP ranks.

    - **dicts** union by key (disjoint per-stage ``tracer.cache()`` keys
      combine; shared keys recurse);
    - **lists** union by position, length-tolerant — shared positions recurse,
      a position only one rank reached is taken as-is, and a trailing no-real
      overshoot tail is dropped; one-sided REAL entries warn (stalled worker /
      divergent control flow) but the complete side is kept;
    - **tuples** of equal length recurse and rebuild (NamedTuple-safe);
    - **leaves** prefer real over sentinel; two reals merge silently when
      equal or emit :class:`PPRankDivergenceWarning` when they differ
      (``label`` names the slot, e.g. ``"noise"`` / ``"outs[2]"``).

    A structural-type clash at one slot (list vs tuple, unequal-length tuples)
    is impossible for model-derived values, so it degrades to the leaf path:
    the mismatch is described, the warning fires, and ``b`` wins — loudly.
    """
    if a is NOT_ON_THIS_RANK:
        return b
    if b is NOT_ON_THIS_RANK:
        return a
    if isinstance(a, list) and isinstance(b, list):
        extra = _real_beyond_common(a, b)
        if extra:
            _warn_divergence(
                label,
                f"one rank produced {extra} real list entr"
                f"{'y' if extra == 1 else 'ies'} the other never reached "
                f"(a stalled or errored worker, or control flow that diverged "
                f"across stages); kept the complete side",
            )
        return _union_sequence(a, b, label)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return _rebuild_sequence(
            a,
            [
                merge_saved(x, y, _extend(label, f"[{i}]"))
                for i, (x, y) in enumerate(zip(a, b))
            ],
        )
    if isinstance(a, CacheView) and isinstance(b, CacheView):
        # Each stage's cache observed its own modules; the union carries every
        # stage's recordings under one view. A path both stages recorded (a
        # module real on every rank) merges entry-wise below.
        a._cache.entries = merge_saved(
            a._cache.entries, b._cache.entries, _extend(label, ".entries")
        )
        return a
    if isinstance(a, Cache) and isinstance(b, Cache):
        a.entries = merge_saved(a.entries, b.entries, _extend(label, ".entries"))
        return a
    if isinstance(a, Entry) and isinstance(b, Entry):
        return Entry(
            output=merge_saved(a.output, b.output, _extend(label, ".output")),
            inputs=merge_saved(a.inputs, b.inputs, _extend(label, ".inputs")),
        )
    if isinstance(a, dict) and isinstance(b, dict):
        # Union the key sets. A normal dict save carries identical keys on
        # every rank (non-owned slots are sentinels), reducing this to the
        # element-wise merge; PP ``tracer.cache()`` keys are DISJOINT per
        # stage, and the union keeps every stage's contributions. ``dict.__*``
        # is used so a dict subclass with overridden lookups can't corrupt the
        # copy.
        merged = {}
        for k in dict.__iter__(a):
            other = (
                dict.__getitem__(b, k) if dict.__contains__(b, k) else NOT_ON_THIS_RANK
            )
            merged[k] = merge_saved(
                dict.__getitem__(a, k), other, _extend(label, f"[{k!r}]")
            )
        for k in dict.__iter__(b):
            if not dict.__contains__(a, k):
                merged[k] = dict.__getitem__(b, k)
        return merged

    detail = _divergence_detail(a, b)
    if detail is not None:
        _warn_divergence(label, detail)
    return b


def merge_shared_saves(mediators: list, per_request_saves: list) -> dict:
    """Merge same-name saves across a multi-invoke trace's requests, in place.

    Locally, invoke blocks written in one frame share their outer names: a
    container bound above the invokes (``rows = [None] * n; rows.save()``) is
    ONE object every invoke mutates. On the engine each invoke rides its own
    request into the worker process with its own COPY, so the copies come back
    one per request, each carrying only that request's writes. A name shipped
    by MORE than one request is such a shared save; this reproduces the
    shared-object semantics by merging its copies element-wise — same-length
    lists slot-wise with ``None`` as the unwritten marker (the common pattern:
    a pre-sized slot list each invoke fills at its own index), dicts by
    key-union, everything else (and any conflicting slot several requests
    wrote) keeping the later request's copy — and writing the merged value
    into every mediator's scope so results push identically whichever mediator
    they're read from. A name only one request shipped stays exactly as that
    request returned it.
    """

    def merge_pair(a, b):
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return [
                merge_pair(x, y) if not (x is None or y is None)
                else (y if x is None else x)
                for x, y in zip(a, b)
            ]
        if isinstance(a, dict) and isinstance(b, dict):
            return {**a, **b}
        return b

    counts: dict = {}
    merged: dict = {}
    for saves in per_request_saves:
        for name, value in saves.items():
            counts[name] = counts.get(name, 0) + 1
            merged[name] = merge_pair(merged[name], value) if name in merged else value
    shared = {name: value for name, value in merged.items() if counts[name] > 1}
    for mediator in mediators:
        for name, value in shared.items():
            mediator.lcls[name] = value
    return shared


def merge_collected(results) -> Optional[dict]:
    """Merge per-rank ``collect_nnsight`` payloads into one collected dict.

    ``results`` is one entry per rank: ``None`` for ranks that ship nothing
    (TP siblings; every rank when nothing was traced), else a pickle of
    ``{engine_id: {"saves": {name: value}, "error": ...}}``. Same-named saves
    merge position-wise with :func:`merge_saved`; the first non-``None`` error
    wins (the same user error raises on every stage — per-rank tracebacks
    differ only in device detail).

    Returns the merged dict, or ``None`` when no rank shipped anything.
    """
    merged: Optional[dict] = None
    for result in results:
        if result is None:
            continue
        collected = pickle.loads(result)
        if merged is None:
            merged = {}
        for engine_id, entry in collected.items():
            dst = merged.setdefault(engine_id, {"saves": {}, "error": None})
            for name, value in entry["saves"].items():
                if name not in dst["saves"]:
                    dst["saves"][name] = value
                else:
                    dst["saves"][name] = merge_saved(
                        dst["saves"][name], value, label=name
                    )
            if dst["error"] is None:
                dst["error"] = entry["error"]
    return merged

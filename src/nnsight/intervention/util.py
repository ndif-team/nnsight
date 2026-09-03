"""Helpers shared across the intervention package.

Two concerns live here, both of them about the boundary between a trace body and
the frame it was written in. The ``input`` helpers give a module's ``(args,
kwargs)`` pair the single-value view users actually write against — ``.input``
reads the first argument and writes it back without the caller having to know the
module's signature. `shared_locals` gives the blocks written in one frame — an
outer trace and the ``tracer.invoke(...)`` blocks inside it — a place to see each
other's names, without letting those names leak into the frame itself: what
escapes a trace is what `save` marked, and nothing else.
"""

from __future__ import annotations

import threading
from types import FrameType
from typing import Any


def first_input(args: tuple, kwargs: dict) -> Any:
    """The first positional argument, or the first keyword one if none positional.

    Raises `StopIteration` for a module called with no arguments at all — there is
    no first input to name. Read ``.inputs`` instead, which is ``((), {})`` there.
    """
    if args:
        return args[0]
    return next(iter(kwargs.values()))


def replace_first_input(args: tuple, kwargs: dict, value: Any) -> tuple:
    """``(args, kwargs)`` with [`first_input`][nnsight.intervention.util.first_input] swapped for ``value``."""
    if args:
        return ((value, *args[1:]), kwargs)
    key = next(iter(kwargs))
    return (args, {**kwargs, key: value})


# Per-frame stores backing `shared_locals`: id(frame) -> (frame, store), cleared
# by the trace that owns the invokes when it finishes
# (`InterleavingTracer.execute`). Thread-local because traces on different threads
# are independent, like the saved set and the nesting depth.
_shared = threading.local()


def shared_locals(frame: FrameType) -> dict:
    """The mapping blocks written in ``frame`` share.

    A name bound by a *sibling* block — an earlier ``tracer.invoke(...)`` — has to
    reach the ones beside it, but must not reach the frame itself: what escapes a
    trace is what `save` marked, and [`push`][nnsight.tracing.util.push] is the
    only thing that decides that.

    Args:
        frame: The frame the blocks were written in.

    Returns:
        The dict those blocks share — the same object for the same frame, until
        the trace that owns them finishes.
    """
    store = getattr(_shared, "store", None)
    if store is None:
        store = _shared.store = {}
    entry = store.get(id(frame))
    if entry is None:
        # The frame is held alongside its store, not just keyed by. A helper that
        # opens an invoke has already returned by the time the trace runs, so its
        # frame would be freed and the next helper allocated at the same address —
        # merging two scopes that the code keeps apart. Frames aren't weak
        # referenceable, so a strong reference until the outermost trace exits is
        # what keeps the id honest.
        entry = store[id(frame)] = (frame, {})
    return entry[1]


def clear_shared_locals() -> None:
    """Drop this thread's per-frame shared stores.

    Called from `InterleavingTracer.execute`, once the trace whose body held the
    invokes is done with them — not at the outermost trace. Each trace body runs
    in a fresh mediator frame, so waiting for the outermost one would hold one
    dead frame per iteration of a loop of traces, and through its locals that
    iteration's tensors, until the session ended.
    """
    store = getattr(_shared, "store", None)
    if store is not None:
        store.clear()

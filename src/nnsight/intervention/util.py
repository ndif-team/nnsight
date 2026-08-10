"""Small helpers shared across the intervention package.
"""

from __future__ import annotations

import threading
from types import FrameType
from typing import Any


def first_input(args: tuple, kwargs: dict) -> Any:
    """The first positional argument, or the first keyword one if none positional."""
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
    invokes is done with them. This used to wait for the *outermost* trace, which
    is fine for one trace and a leak for a loop of them: each trace body runs in a
    fresh mediator frame, so every iteration added an entry holding that frame —
    and, through its locals, that iteration's tensors — until the session ended.
    """
    store = getattr(_shared, "store", None)
    if store is not None:
        store.clear()

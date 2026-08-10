"""Helpers for the tracer: frame serialization, traceback cleanup, and writing
variables back into a live frame.

These support two jobs that the [`Tracer`][nnsight.tracing.tracer.Tracer] can't do
inline:

- **Serialization** — a real Python frame can't be pickled, so
  [`SerializedFrame`][nnsight.tracing.util.SerializedFrame] stands in for one when a trace is shipped to a remote
  worker, keeping just the code metadata needed for tracebacks.
- **Traceback surgery** — user intervention code runs deep inside the model's
  forward pass, so exceptions come back wrapped in nnsight frames.
  [`clean_traceback`][nnsight.tracing.util.clean_traceback] strips nnsight internals away so the error points at
  the user's own code (across whatever files it spans).
- **Frame write-back** — [`push`][nnsight.tracing.util.push] copies the results of an executed block
  back into the frame the ``with`` statement lived in, so assignments made inside
  the block are visible afterwards.
- **Block scope** — [`Scope`][nnsight.tracing.util.Scope] is the namespace a captured block runs in, and
  the reason a block can read the names around it even though it runs later, and
  somewhere else, than the line it was written on.
"""

from __future__ import annotations

import ctypes
import os
import sys
import types
from types import FrameType, TracebackType
from typing import Any

# Flushes a frame's f_locals dict into its fast-locals array; removed in 3.13,
# where f_locals writes through on its own (PEP 667). Declared with an explicit
# signature so ctypes doesn't have to guess at the call site. See `push`.
if sys.version_info < (3, 13):
    _locals_to_fast = ctypes.pythonapi.PyFrame_LocalsToFast
    _locals_to_fast.argtypes = (ctypes.py_object, ctypes.c_int)
    _locals_to_fast.restype = None
else:
    _locals_to_fast = None

class Scope(dict):
    """The namespace a captured block runs in.

    A block runs later than the line it was written on — and, once interleaving,
    somewhere else entirely — so the names it reaches have to come from three
    places, in order:

    1. **This dict**, a snapshot of the frame's locals taken when the block was
       captured. A name the block only reads means what it meant *where the block
       was written*: a ``for prompt in prompts:`` variable has moved on (or gone)
       by the time the block runs, so the snapshot is what keeps it right.
    2. [`shared`][nnsight.tracing.util.Scope.shared] — the store the blocks written in that frame share
       ([`shared_locals`][nnsight.intervention.util.shared_locals]). A name bound by a
       *sibling* block (an earlier ``tracer.invoke(...)``) isn't in the snapshot,
       because nothing had bound it when this block was captured. Blocks written in
       one frame share these; blocks written in different functions have different
       frames, so they stay as separate as their code.
    3. `glbls` — the frame's globals, reached by fallback rather than copied.

    Passing this as ``exec``'s *globals* (not just its locals) is what lets a
    ``lambda`` or nested ``def`` inside a block reach the block's own names: their
    free variables compile to ``LOAD_GLOBAL``, which never consults a locals
    mapping — but does honor ``__missing__`` on a dict subclass.

    Writes land here *and* in [`shared`][nnsight.tracing.util.Scope.shared], so the block sees its own
    assignments and so do the blocks written beside it — but not the frame, which
    only [`push`][nnsight.tracing.util.push] writes to. Because only the snapshot
    and the block's writes are ever stored, iterating a scope yields the block's
    own names — what [`push`][nnsight.tracing.util.push] and the block reducers want — and never the
    whole global namespace.
    """

    def __init__(self, own: dict, shared: dict, glbls: dict) -> None:
        super().__init__(own)
        self.shared = shared
        self.glbls = glbls

    def __missing__(self, key: str) -> Any:
        try:
            return self.shared[key]
        except KeyError:
            return self.glbls[key]

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self.shared[key] = value

    def copy(self) -> "Scope":
        """A scope with this one's names, sharing the same frame and globals."""
        return Scope(dict(self), self.shared, self.glbls)


class SerializedFrame:
    """A picklable stand-in for a real frame.

    Carries only the code metadata (filename/lineno/name) needed remotely for
    tracebacks and source lookup; f_locals/f_globals are empty because the
    replay closure already captured the real ones.
    """

    def __init__(self, co_filename: str, co_firstlineno: int, co_name: str) -> None:
        self.f_locals: dict = {}
        self.f_globals: dict = {}
        self.f_code = types.SimpleNamespace(
            co_filename=co_filename,
            co_firstlineno=co_firstlineno,
            co_name=co_name,
        )

    @classmethod
    def of(cls, frame: FrameType) -> SerializedFrame:
        """Build a serializable stand-in from a live ``frame``."""
        code = frame.f_code
        return cls(code.co_filename, code.co_firstlineno, code.co_name)

# The nnsight package root; frames from inside it are stripped from tracebacks
# of exceptions raised by user code running under a tracer.
_INTERNAL = os.path.dirname(os.path.dirname(__file__))


def filter_traceback(
    traceback: TracebackType | None, keep
) -> TracebackType | None:
    """Rebuild a traceback keeping only frames for which ``keep(frame)`` is true.

    Tracebacks are immutable and singly linked, so we can't delete entries in
    place — we collect the ones to keep and re-link a fresh chain (from the tail
    up, since each `TracebackType` points to the *next* frame).

    Args:
        traceback: The head of the traceback chain to filter (may be ``None``).
        keep: Predicate called with each frame; the entry is kept when it returns
            true.

    Returns:
        The head of the filtered chain, or ``None`` if nothing was kept.
    """
    kept = []
    while traceback is not None:
        if keep(traceback.tb_frame):
            kept.append(traceback)
        traceback = traceback.tb_next

    result = None
    for entry in reversed(kept):
        result = TracebackType(result, entry.tb_frame, entry.tb_lasti, entry.tb_lineno)
    return result


def clean_traceback(traceback: TracebackType | None) -> TracebackType | None:
    """Drop nnsight-internal frames, leaving only the user's frames.

    Used as the fallback when we can't pin the error to the traced block's own
    file: at least hide the library's plumbing. In debug mode
    (``CONFIG.APP.DEBUG``) nothing is stripped, so the full stack — nnsight
    internals included — is shown.
    """
    from ..schema.config import CONFIG

    if CONFIG.APP.DEBUG:
        return traceback
    return filter_traceback(
        traceback, lambda frame: not frame.f_code.co_filename.startswith(_INTERNAL)
    )


def push(frame: FrameType, variables: dict[str, Any]) -> None:
    """Write ``variables`` back into a live ``frame``'s locals.

    The traced block runs in a scratch namespace, not the original frame, so its
    results have to be copied back for assignments to be visible after the
    ``with`` statement.

    On Python < 3.13, ``frame.f_locals`` is a snapshot and updating it doesn't
    reach the interpreter's fast-locals array, so we call the C-API
    ``PyFrame_LocalsToFast`` to flush the change through. From 3.13 on,
    ``f_locals`` is a live write-through mapping (PEP 667) and the plain
    ``update`` is enough.

    This is the *only* route from a block's namespace back into the frame — see
    [`shared_locals`][nnsight.intervention.util.shared_locals], which keeps a block's
    other writes out of it.

    Args:
        frame: The frame to write into (the one the ``with`` block lived in), or a
            [`SerializedFrame`][nnsight.tracing.util.SerializedFrame] standing in
            for one, whose ``f_locals`` is an ordinary dict.
        variables: Names and values to set on that frame.
    """
    frame.f_locals.update(variables)
    # A SerializedFrame is not a PyFrameObject: its plain-dict update is the whole
    # job, and handing it to the C API dereferences it as a frame (segfault on
    # 3.10, silent undefined behavior on 3.11/3.12).
    if _locals_to_fast is not None and isinstance(frame, FrameType):
        _locals_to_fast(ctypes.py_object(frame), ctypes.c_int(1))

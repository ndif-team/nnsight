"""Process-wide caches shared by every [`Tracer`][nnsight.tracing.tracer.Tracer].

Tracing a ``with`` block means finding its source, parsing it to an AST, and
compiling the block body — work that is identical every time the *same* block is
entered. These two module-level dicts memoize that work across all tracer
instances so, e.g., a ``with model.trace(...)`` inside a Python loop is read,
parsed, and compiled exactly once.

Both grow with the number of distinct source files and trace sites seen during
the process, which is bounded by the program's own source, so they are never
evicted. Neither is re-validated against on-disk changes: a source edited
mid-run is still traced as it was first seen (see [`Tracer.source`][nnsight.tracing.tracer.Tracer.source]).
A process that builds blocks it receives — a server — runs other programs'
source, so it confines each build to itself with `block_scope`.
"""

from __future__ import annotations

import contextlib
import linecache
from ast import AST
from types import CodeType
from typing import Iterator

#: Raw source text keyed by filename, read once and reused. Populated lazily by
#: [`source`][nnsight.tracing.tracer.Tracer.source].
SOURCES: dict[str, str] = {}

#: Parsed + compiled ``with`` blocks, memoized per trace site.
#:
#: Key: ``(filename, lineno, co_name, co_firstlineno)`` — enough to pin down a
#: single ``with`` statement in the program.
#: Value: ``(node, code)`` — the block's AST node and its compiled body — or
#: ``(None, None)`` for a site that turned out not to be a ``with`` block (that
#: negative verdict is cached too, so repeated non-block captures don't reparse).
BLOCKS: dict[tuple[str, int, str, int], tuple[AST | None, CodeType | None]] = {}


@contextlib.contextmanager
def block_scope() -> Iterator[None]:
    """Undo, on exit, what building a received block added to these caches and ``linecache``.

    [`RequestModel.deserialize`][nnsight.schema.request.RequestModel.deserialize]
    registers a block's source in ``linecache`` under the client's filename, and
    the invokes inside it are parsed from there and memoized here under that
    filename. Kept, a later request from the same file is parsed against its
    predecessor's block — running the old invoke body, or finding no ``with`` at
    its line — and on a long-lived server all three grow with every client file
    seen.

    The restore undoes whatever else registered meanwhile, so builds under it
    must not interleave.
    """
    lines, sources, blocks = dict(linecache.cache), dict(SOURCES), dict(BLOCKS)
    try:
        yield
    finally:
        linecache.cache.clear()
        linecache.cache.update(lines)
        SOURCES.clear()
        SOURCES.update(sources)
        BLOCKS.clear()
        BLOCKS.update(blocks)

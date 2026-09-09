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
"""

from __future__ import annotations

from ast import AST
from types import CodeType

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

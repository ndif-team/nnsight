"""A static type for values read inside a trace.

Inside a ``with model.trace(...)`` block, reading ``envoy.output`` (or
``.input``, a ``source`` op, ...) yields a value that behaves like the underlying
activation — you can index it, call tensor methods on it, and ``.save()`` it to
keep it past the block. At runtime that value simply *is* the activation (a real
``torch.Tensor``, tuple, ...) with ``.save()`` mounted on every object.

[`Object`][nnsight.tracing.hint.Object] exists only to give that value a name for **type hints**: it looks
like a ``torch.Tensor`` (so editors offer tensor methods and indexing) and carries
a ``.save()`` returning itself. It is never really instantiated — annotate
trace-time values with it (``def output(self) -> Object``) so callers get useful
completions.
"""

from __future__ import annotations

from typing import Any

import torch

try:  # 3.11+: a precise self-type; fall back to the class name pre-3.11.
    from typing import Self
except ImportError:  # pragma: no cover
    Self = "Object"


class Object(torch.Tensor):
    """Tensor-like stand-in for a value read during a trace (hint use only)."""

    def save(self, _: Any = 0) -> Self:
        """Keep this value accessible after the trace block exits."""
        from .tracer import mark

        # mark, not save: an Object is a static hint stand-in, used to describe
        # values outside a running trace, so it skips save()'s in-a-trace guard.
        mark(self)
        return self

    def __getattr__(self, name: str) -> Self:
        return super().__getattr__(name)

    def __getitem__(self, key: Any) -> Self:
        return super().__getitem__(key)

    def __call__(self, *args: Any, **kwargs: Any) -> Self:
        return super().__call__(*args, **kwargs)

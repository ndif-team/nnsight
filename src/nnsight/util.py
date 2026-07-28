from __future__ import annotations

import importlib
from typing import Any, Callable


def apply(data: Any, fn: Callable[[Any], Any], cls: type, n: int = 0) -> Any:
    """Recursively apply ``fn`` to every instance of ``cls`` found in ``data``.

    Containers — ``list``, ``tuple`` (incl. namedtuples), ``dict``, ``set``,
    ``frozenset`` and ``slice`` — are traversed without limit and rebuilt with
    the transformed values (the originals are left untouched). Generic objects
    (anything exposing a ``__dict__``) are also descended into, with their
    attributes transformed and written back *in place*, but only up to ``n``
    levels of object nesting so we never recurse without bound through an
    arbitrary object graph.

    Container nesting is free — it does not count against ``n``. Only stepping
    into a generic object spends a level.

    Args:
        data: The structure to traverse.
        fn: Called on each ``cls`` instance; its return value replaces it.
        cls: The leaf type to transform.
        n: Maximum depth of generic-object descent. ``n=0`` (default) disables
            object descent entirely — only containers are traversed. ``n=1``
            descends into the first object encountered along a path but not into
            objects nested within it, and so on.

    Returns:
        The transformed structure. Containers are new objects; generic objects
        (when descended into) are mutated in place and returned as-is.
    """
    if isinstance(data, cls):
        return fn(data)

    if isinstance(data, list):
        return type(data)(apply(value, fn, cls, n) for value in data)

    if isinstance(data, tuple):
        items = [apply(value, fn, cls, n) for value in data]
        # namedtuples take positional fields rather than a single iterable.
        return type(data)(*items) if hasattr(data, "_fields") else type(data)(items)

    if isinstance(data, dict):
        # Copy first to preserve the exact type (OrderedDict, defaultdict, ...)
        # without tripping over subclass constructors.
        new = data.copy()
        for key, value in data.items():
            new[key] = apply(value, fn, cls, n)
        return new

    if isinstance(data, (set, frozenset)):
        return type(data)(apply(value, fn, cls, n) for value in data)

    if isinstance(data, slice):
        return slice(
            apply(data.start, fn, cls, n),
            apply(data.stop, fn, cls, n),
            apply(data.step, fn, cls, n),
        )

    # Generic object: descend into its attributes in place, budget permitting.
    if n > 0 and hasattr(data, "__dict__"):
        for key, value in vars(data).items():
            transformed = apply(value, fn, cls, n - 1)
            if transformed is not value:
                setattr(data, key, transformed)
        return data

    return data


def to_import_path(obj: type) -> str:
    """The dotted ``module.QualName`` path that [`from_import_path`][nnsight.util.from_import_path] resolves."""
    return f"{obj.__module__}.{obj.__qualname__}"


def from_import_path(path: str) -> Any:
    """Resolve a dotted ``module.QualName`` path (as produced by [`to_import_path`][nnsight.util.to_import_path])."""
    module_path, _, qualname = path.rpartition(".")
    module = importlib.import_module(module_path)
    obj = module
    for name in qualname.split("."):
        obj = getattr(obj, name)
    return obj

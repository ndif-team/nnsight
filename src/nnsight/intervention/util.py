"""Small helpers shared across the intervention package."""

from __future__ import annotations

from typing import Any


def first_input(args: tuple, kwargs: dict) -> Any:
    """The first positional argument, or the first keyword one if none positional."""
    if args:
        return args[0]
    return next(iter(kwargs.values()))


def replace_first_input(args: tuple, kwargs: dict, value: Any) -> tuple:
    """``(args, kwargs)`` with :func:`first_input` swapped for ``value``."""
    if args:
        return ((value, *args[1:]), kwargs)
    key = next(iter(kwargs))
    return (args, {**kwargs, key: value})

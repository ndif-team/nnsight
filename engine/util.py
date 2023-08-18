import time
from functools import wraps
from typing import Any, Callable, Type, Union

import torch

Primative = Union[str, int, float, bool]
Value = Union[Primative, torch.Tensor]


def apply(data: Any, fn: Callable, cls: type):
    if isinstance(data, cls):
        return fn(data)

    if isinstance(data, list):
        return [apply(_data, fn, cls) for _data in data]

    if isinstance(data, tuple):
        return tuple([apply(_data, fn, cls) for _data in data])

    if isinstance(data, dict):
        return {key: apply(value, fn, cls) for key, value in data.items()}

    return data

def fetch_attr(object:object, target: str):
    target_atoms = target.split(".")
    for i, atom in enumerate(target_atoms):
        if not hasattr(object, atom):
            return None
        object = getattr(object, atom)
    return object

def wrap(object:object, wrapper:Type, *args, **kwargs):
   
    if isinstance(object, wrapper):
        return object

    object.__class__ = type(
        object.__class__.__name__, (wrapper, object.__class__), {}
    )

    wrapper.__init__(object, *args, **kwargs)

    return object

def timed(func, lggr):
    """This decorator prints the execution time for the decorated function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        lggr.debug(f"Method `{func.__qualname__}` ran in {round(end - start, 6)}s")
        return result

    return wrapper

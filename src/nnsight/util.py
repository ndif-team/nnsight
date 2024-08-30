"""Module for utility functions and classes used throughout the package."""

import importlib
import types
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch

if TYPE_CHECKING:
    from .tracing.Node import Node

# TODO Have an Exception you can raise to stop apply early

def apply(
    data: Any, fn: Callable, cls: Type, inplace: bool = False
) -> Collection:
    """Applies some function to all members of a collection of a give type (or types)

    Args:
        data (Any): Collection of data to apply function to.
        fn (Callable): Function to apply.
        cls (type): Type or Types to apply function to.
        inplace (bool): If to apply the fn inplace. (For lists and dicts)

    Returns:
        Any: Same kind of collection as data, after then fn has been applied to members of given type.
    """

    if isinstance(data, cls):
        return fn(data)

    data_type = type(data)

    if data_type == list:
        if inplace:
            for idx, _data in enumerate(data):
                data[idx] = apply(_data, fn, cls, inplace=inplace)
            return data
        return [apply(_data, fn, cls, inplace=inplace) for _data in data]

    elif data_type == tuple:
        return tuple([apply(_data, fn, cls, inplace=inplace) for _data in data])

    elif data_type == dict:
        if inplace:
            for key, value in data.items():
                data[key] = apply(value, fn, cls, inplace=inplace)
            return data
        return {
            key: apply(value, fn, cls, inplace=inplace)
            for key, value in data.items()
        }

    elif data_type == slice:
        return slice(
            apply(data.start, fn, cls, inplace=inplace),
            apply(data.stop, fn, cls, inplace=inplace),
            apply(data.step, fn, cls, inplace=inplace),
        )

    return data


def fetch_attr(object: object, target: str) -> Any:
    """Retrieves an attribute from an object hierarchy given an attribute path. Levels are separated by '.' e.x (transformer.h.1)

    Args:
        object (object): Root object to get attribute from.
        target (str): Attribute path as '.' separated string.

    Returns:
        Any: Fetched attribute.
    """
    if target == "":
        return object

    target_atoms = target.split(".")

    for atom in target_atoms:

        if not atom:
            continue

        object = getattr(object, atom)

    return object


def wrap(object: object, wrapper: Type, *args, **kwargs) -> object:
    """Wraps some object given some wrapper type.
    Updates the __class__ attribute of the object and calls the wrapper type's __init__ method.

    Args:
        object (object): Object to wrap.
        wrapper (Type): Type to wrap the object in.

    Returns:
        object: Wrapped object.
    """
    if isinstance(object, wrapper):
        return object

    new_class = types.new_class(
        object.__class__.__name__,
        (object.__class__, wrapper),
    )

    object.__class__ = new_class

    wrapper.__init__(object, *args, **kwargs)

    return object


def to_import_path(type: type) -> str:

    return f"{type.__module__}.{type.__name__}"


def from_import_path(import_path: str) -> type:

    *import_path, classname = import_path.split(".")
    import_path = ".".join(import_path)

    return getattr(importlib.import_module(import_path), classname)


class WrapperModule(torch.nn.Module):
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args

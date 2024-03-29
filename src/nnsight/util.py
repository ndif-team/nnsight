"""Module for utility functions and classes used throughout the package."""

import time
import types
from functools import wraps
from typing import Any, Callable, Collection, Type

import torch


def apply(data: Collection, fn: Callable, cls: Type) -> Collection:
    """Applies some function to all members of a collection of a give type (or types)

    Args:
        data (Collection): Collection to apply function to.
        fn (Callable): Function to apply.
        cls (type): Type or Types to apply function to.

    Returns:
        Collection: Same kind of collection as data, after then fn has been applied to members of given type.
    """
    if isinstance(data, cls):
        return fn(data)

    data_type = type(data)

    if data_type == list:
        return [apply(_data, fn, cls) for _data in data]

    if data_type == tuple:
        return tuple([apply(_data, fn, cls) for _data in data])

    if data_type == dict:
        return {key: apply(value, fn, cls) for key, value in data.items()}

    if data_type == slice:
        return slice(
            apply(data.start, fn, cls),
            apply(data.stop, fn, cls),
            apply(data.step, fn, cls),
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


def meta_deepcopy(self: torch.nn.parameter.Parameter, memo):
    if id(self) in memo:
        return memo[id(self)]
    else:
        result = type(self)(
            torch.empty_like(self.data, dtype=self.data.dtype, device="meta"),
            self.requires_grad,
        )
        memo[id(self)] = result
        return result


class WrapperModule(torch.nn.Module):
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args

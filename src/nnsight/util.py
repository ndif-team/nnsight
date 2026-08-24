"""Module for utility functions and classes used throughout the package."""

import copy
import importlib
from contextlib import AbstractContextManager
from typing import Any, Callable, Collection, List, Optional, Type, TypeVar

import torch
from typing_extensions import Self

# TODO Have an Exception you can raise to stop apply early

T = TypeVar("T")
C = TypeVar("C", bound=Collection[T])


def _rebuild_tuple(data: tuple, items: List[Any]) -> tuple:
    """Rebuild a ``tuple`` subclass from *items*, preserving its concrete type.

    ``tuple`` subclasses do not share a constructor signature: ``namedtuple``
    takes one argument per field, while ``torch.Size`` and the
    ``torch.return_types.*`` structsequences take a single iterable. Try the
    type-appropriate constructor and fall back to a plain ``tuple`` only if the
    subclass cannot be rebuilt at all.
    """

    if hasattr(data, "_make"):
        # namedtuple
        return type(data)._make(items)

    try:
        # torch.Size, torch.return_types.*, and tuple subclasses that accept an
        # iterable.
        return type(data)(items)
    except TypeError:
        return tuple(items)


def _apply_mapping(
    data: dict, fn: Callable[[T], Any], cls: Type[T], inplace: bool
) -> dict:
    """Apply *fn* through a ``dict`` subclass, preserving its concrete type.

    A shallow copy is mutated rather than the type being reconstructed from its
    items, because subclasses such as HuggingFace's ``ModelOutput`` are
    dataclasses whose ``__init__`` does not accept a mapping. Assigning through
    ``__setitem__`` also keeps subclass invariants intact -- ``ModelOutput``, for
    instance, mirrors every item onto the matching attribute, so
    ``output.logits`` stays in sync with ``output["logits"]``.
    """

    new = data if inplace else copy.copy(data)

    for key in list(data.keys()):
        new[key] = apply(data[key], fn, cls, inplace=inplace)

    return new


def _apply_sequence(
    data: list, fn: Callable[[T], Any], cls: Type[T], inplace: bool
) -> list:
    """Apply *fn* through a ``list`` subclass, preserving its concrete type."""

    new = data if inplace else copy.copy(data)

    for idx in range(len(data)):
        new[idx] = apply(data[idx], fn, cls, inplace=inplace)

    return new


def apply(data: C, fn: Callable[[T], Any], cls: Type[T], inplace: bool = False) -> C:
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
            key: apply(value, fn, cls, inplace=inplace) for key, value in data.items()
        }

    elif data_type == slice:
        return slice(
            apply(data.start, fn, cls, inplace=inplace),
            apply(data.stop, fn, cls, inplace=inplace),
            apply(data.step, fn, cls, inplace=inplace),
        )

    # The checks above match the builtin container types exactly, which leaves
    # every *subclass* of them untouched. That silently skips containers nnsight
    # handles constantly -- a HuggingFace `ModelOutput` (an `OrderedDict`
    # subclass) is what every top-level transformers module returns, and
    # `namedtuple` / `torch.Size` / `torch.return_types.*` are ordinary module
    # outputs too. Recurse into them as well, rebuilding the concrete type so
    # callers still get a `ModelOutput` back (`.logits`, `.last_hidden_state`,
    # ...) rather than a plain dict.
    elif isinstance(data, dict):
        return _apply_mapping(data, fn, cls, inplace)

    elif isinstance(data, list):
        return _apply_sequence(data, fn, cls, inplace)

    elif isinstance(data, tuple):
        return _rebuild_tuple(
            data, [apply(_data, fn, cls, inplace=inplace) for _data in data]
        )

    return data


def applyn(data: C, fn: Callable[[T], Any], cls: Type[T], inplace: bool = False) -> C:
    """Applies some function to all members of a collection of a give type (or types)

    Args:
        data (Any): Collection of data to apply function to.
        fn (Callable): Function to apply.
        cls (type): Type or Types to apply function to.
        inplace (bool): If to apply the fn inplace. (For lists and dicts)

    Returns:
        Any: Same kind of collection as data, after then fn has been applied to members of given type.
    """

    if isinstance(data[0], cls):
        return fn(*data)

    data_type = type(data[0])

    if data_type == list:
        # if inplace:
        #     for idx, _data in enumerate(data):
        #         data[idx] = apply(_data, fn, cls, inplace=inplace)
        #     return data

        return [
            applyn([_data[i] for _data in data], fn, cls, inplace=inplace)
            for i in range(len(data[0]))
        ]

    elif data_type == tuple:
        return tuple(
            [
                applyn([_data[i] for _data in data], fn, cls, inplace=inplace)
                for i in range(len(data[0]))
            ]
        )

    elif data_type == dict:
        # if inplace:
        #     for key, value in data.items():
        #         data[key] = apply(value, fn, cls, inplace=inplace)
        #     return data
        return {
            key: applyn([_data[key] for _data in data], fn, cls, inplace=inplace)
            for key in data[0].keys()
        }

    # elif data_type == slice:
    #     return slice(
    #         apply(data.start, fn, cls, inplace=inplace),
    #         apply(data.stop, fn, cls, inplace=inplace),
    #         apply(data.step, fn, cls, inplace=inplace),
    #     )

    # Container subclasses -- see the matching comment in `apply`. Without these
    # branches a batched write into a `ModelOutput` replaces nothing at all.
    elif isinstance(data[0], dict):

        new = copy.copy(data[0])

        for key in list(data[0].keys()):
            # A sibling may legitimately lack a key: `ModelOutput.keys()` only
            # reports its non-`None` fields, so a user-supplied replacement need
            # not carry every field the original has. Leave those alone rather
            # than raising.
            if all(key in _data for _data in data[1:]):
                new[key] = applyn(
                    [_data[key] for _data in data], fn, cls, inplace=inplace
                )

        return new

    elif isinstance(data[0], (list, tuple)):

        items = [
            applyn([_data[i] for _data in data], fn, cls, inplace=inplace)
            for i in range(len(data[0]))
        ]

        if isinstance(data[0], tuple):
            return _rebuild_tuple(data[0], items)

        new = copy.copy(data[0])

        for i, item in enumerate(items):
            new[i] = item

        return new

    return data[0]


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


def to_import_path(type: type) -> str:

    return f"{type.__module__}.{type.__name__}"


def from_import_path(import_path: str) -> type:

    *import_path, classname = import_path.split(".")
    import_path = ".".join(import_path)

    return getattr(importlib.import_module(import_path), classname)


class Patch:
    """Class representing a replacement of an attribute on a module.

    Attributes:
        obj (Any): Object to replace.
        replacement (Any): Object that replaces.
        parent (Any): Module or class to replace attribute.
    """

    def __init__(
        self,
        parent: Any,
        replacement: Any = None,
        key: str = None,
        as_dict: bool = False,
    ) -> None:
        self.parent = parent
        self.replacement = replacement
        self.key = key

        self.as_dict = as_dict

        if self.as_dict:
            self.orig = self.parent[key]
        else:
            self.orig = getattr(self.parent, key)

    def patch(self) -> None:
        """Carries out the replacement of an object in a module/class."""
        if self.replacement is None:
            if self.as_dict:
                del self.parent[self.key]
            else:
                delattr(self.parent, self.key)
        else:
            if self.as_dict:
                self.parent[self.key] = self.replacement
            else:
                setattr(self.parent, self.key, self.replacement)

    def restore(self) -> None:
        """Carries out the restoration of the original object on the objects module/class."""

        try:
            if self.as_dict:
                self.parent[self.key] = self.orig
            else:
                setattr(self.parent, self.key, self.orig)
        except Exception as e:
            pass


class Patcher(AbstractContextManager):
    """Context manager that patches from a list of Patches on __enter__ and restores the patch on __exit__.

    Attributes:
        patches (List[Patch]):
    """

    def __init__(self, patches: Optional[List[Patch]] = None) -> None:
        self.patches = patches or []

        self.entered = False

    def add(self, patch: Patch) -> None:
        """Adds a Patch to the patches. Also calls `.patch()` on the Patch.

        Args:
            patch (Patch): Patch to add.
        """
        self.patches.append(patch)

        if self.entered:
            patch.patch()

    def __enter__(self) -> Self:
        """Enters the patching context. Calls `.patch()` on all patches.

        Returns:
            Patcher: Patcher
        """
        if not self.entered:
            self.entered = True

            for patch in self.patches:
                patch.patch()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Calls `.restore()` on all patches."""

        if self.entered:
            self.entered = False
            for patch in self.patches:
                patch.restore()


class WrapperModule(torch.nn.Module):

    def forward(self, *args, **kwargs):

        if len(args) == 1:
            return args[0]

        return args, kwargs

"""Module for utility functions and classes used throughout the package."""

import importlib
from contextlib import AbstractContextManager
from typing import Any, Callable, Collection, List, Optional, Type, TypeVar

import torch
from typing_extensions import Self

# TODO Have an Exception you can raise to stop apply early

T = TypeVar("T")
C = TypeVar("C", bound=Collection[T])


def apply(
    data: C, fn: Callable[[T], Any], cls: Type[T], inplace: bool = False
) -> C:
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

    def __init__(self, parent: Any, replacement: Any, key: str) -> None:
        self.parent = parent
        self.replacement = replacement
        self.key = key
        self.orig = getattr(self.parent, key)

    def patch(self) -> None:
        """Carries out the replacement of an object in a module/class."""
        setattr(self.parent, self.key, self.replacement)

    def restore(self) -> None:
        """Carries out the restoration of the original object on the objects module/class."""

        setattr(self.parent, self.key, self.orig)

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
        
        self.entered = True
        
        for patch in self.patches:
            patch.patch()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Calls `.restore()` on all patches."""
        self.entered = False
        for patch in self.patches:
            patch.restore()


class WrapperModule(torch.nn.Module):
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args

class NNsightError(Exception):
    """NNsight Execption class for raising error during execution.
    
    Attributes:
        - message (str): error message.
        - node_id (int): node id.
        - traceback_content (Optional[str]): traceback of the original exception being raised.
    """

    def __init__(self, message: str, node_id: int, traceback_content: Optional[str] = None):
        self.message = message
        self.node_id = node_id
        self.traceback_content = traceback_content
        super().__init__(self.message)


    def _render_traceback_(self) -> List[str]:
        """
        This function allows custom rendering of traceback in IPython
        
        Returns:
            - List of string lines.
        """
        traceback_list = self.traceback_content.split("\n")
        traceback_list.append(f"{str(self.__class__.__name__)}: {self.message}")

        return traceback_list

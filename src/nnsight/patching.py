"""The patching module handles patching of classes and functions in modules."""
from __future__ import annotations

import importlib
import types
from contextlib import AbstractContextManager
from typing import Any, List

from . import util


class Patch:
    """Class representing a replacement of an attribute on a module.

    Attributes:
        obj (Any): Object to replace.
        replacement (Any): Object that replaces.
        parent (Any): Module or class to replace attribute.
    """

    def __init__(self, obj: Any, replacement: Any) -> None:
        self.obj = obj
        self.replacement = replacement

        builtin: bool = isinstance(self.obj, types.BuiltinFunctionType)
        module = importlib.import_module(self.obj.__module__)

        if builtin:
            self.parent = module
        else:
            parent_path = ".".join(self.obj.__qualname__.split(".")[:-1])

            self.parent = (
                util.fetch_attr(module, parent_path) if parent_path else module
            )

    def patch(self) -> None:
        """Carries out the replacement of an object in a module/class.

        Imports the objects module with:

        .. code-block:: python

            importlib.import_module(self.obj.__module__)

        And replaces it with:

        .. code-block:: python

            setattr(module, self.obj.__name__, self.replacement)

        """
        setattr(self.parent, self.obj.__name__, self.replacement)

    def restore(self) -> None:
        """Carries out the restoration of the original object on the objects module/class.

        Imports the objects module with:

        .. code-block:: python

            importlib.import_module(self.obj.__module__)

        And replaces it with:

        .. code-block:: python

            setattr(module, self.obj.__name__, self.obj)

        """

        setattr(self.parent, self.obj.__name__, self.obj)


class Patcher(AbstractContextManager):
    """Context manager that patches from a list of Patches on __enter__ and restores the patch on __exit__.

    Attributes:
        patches (List[Patch]):
    """

    def __init__(self, patches: List[Patch] = None) -> None:
        self.patches = patches or []

    def add(self, patch: Patch) -> None:
        """Adds a Patch to the patches. Also calls `.patch()` on the Patch.

        Args:
            patch (Patch): Patch to add.
        """
        self.patches.append(patch)

        patch.patch()

    def __enter__(self) -> Patcher:
        """Enters the patching context. Calls `.patch()` on all patches.

        Returns:
            Patcher: Patcher
        """
        for patch in self.patches:
            patch.patch()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Calls `.restore()` on all patches."""
        for patch in self.patches:
            patch.restore()

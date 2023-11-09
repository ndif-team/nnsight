"""The patching module handles patching of classes and functions in modules.

Attributes:
    DEFAULT_PATCHER (Patcher): The default patcher that patches some torch functions on initialization. 

"""
from __future__ import annotations

import importlib
from contextlib import AbstractContextManager
from typing import Any, List


class Patch:
    """Class representing a replacement of an attribute on a module.

    Attributes:
        obj (Any): Object to replace.
        replacement (Any): Object that replaces.
    """

    def __init__(self, obj: Any, replacement: Any) -> None:
        self.obj = obj
        self.replacement = replacement

    def patch(self) -> None:
        """Carries out the replacement of an object in a module.

        Imports the objects module with:
        
        .. code-block:: python

            importlib.import_module(self.obj.__module__)

        And replaces it with:

        .. code-block:: python
            
            setattr(module, self.obj.__name__, self.replacement)

        """
        module = importlib.import_module(self.obj.__module__)

        setattr(module, self.obj.__name__, self.replacement)

    def restore(self) -> None:
        """Carries out the restoration of the original object on the objects module.

        Imports the objects module with:

        .. code-block:: python

            importlib.import_module(self.obj.__module__)
            
        And replaces it with:

        .. code-block:: python

            setattr(module, self.obj.__name__, self.obj)

        """
        module = importlib.import_module(self.obj.__module__)

        setattr(module, self.obj.__name__, self.obj)


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


DEFAULT_PATCHER = Patcher()

from functools import wraps

import torch


def repeat_interleave_wrapper(fn):
    @wraps(fn)
    def repeat_interleave(
        input: torch.Tensor, repeats: torch.LongTensor, dim=None, output_size=None
    ):
        if input.device.type == "meta":
            if not isinstance(repeats, torch.Tensor):
                repeats = torch.LongTensor([repeats])

            if dim is None:
                input = input.flatten()
                dim = 0

            if repeats.dim() == 0 or (repeats.dim() == 1 and repeats.size(0) == 1):
                repeats = repeats.reshape([1]).expand([input.size(dim)])

            new_dim_size = repeats.cumsum(0)[-1].item()
            new_output_shape = list(input.shape)
            new_output_shape[dim] = new_dim_size

            return torch.empty(new_output_shape, device="meta")

        else:
            return fn(input, repeats, dim=dim, output_size=output_size)

    return repeat_interleave


DEFAULT_PATCHER.add(
    Patch(torch.repeat_interleave, repeat_interleave_wrapper(torch.repeat_interleave))
)


DEFAULT_PATCHER.__enter__()

from torch._meta_registrations import register_meta, aten, global_decomposition_table, _meta_lib_dont_use_me_use_register_meta

def activate_recent_meta():
    op_overload, fn = list(global_decomposition_table['meta'].items())[-1]
    op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)
    _meta_lib_dont_use_me_use_register_meta.impl(op_overload, fn)



@register_meta(aten._local_scalar_dense)
def local_scalar_dense_meta(A):

    return 0

activate_recent_meta()




import operator
from inspect import (
    getmembers,
    isbuiltin,
    isclass,
    isfunction,
    ismethod,
    ismethoddescriptor,
)
from typing import Callable, Union

import einops
import torch
from torch.utils.data.dataloader import DataLoader

from ... import util
from ...intervention import protocols as intervention_protocols
from ...tracing import protocols
from ...tracing.graph import Proxy
from ...tracing import contexts
from ...intervention import contexts as intervention_contexts

def get_function_name(fn, module_name=None):
    if isinstance(fn, str):
        return fn

    if module_name is not None:
        return f"{module_name}.{fn.__name__}"

    module_name = getattr(fn, "__module__", None)

    if module_name is None:
        return fn.__qualname__

    return f"{module_name}.{fn.__qualname__}"


def update_function(function: Union[str,  Callable], new_function: Callable):

    if not isinstance(function, str):

        function = get_function_name(function)

    new_function.__name__ = function

    FUNCTIONS_WHITELIST[function] = new_function


FUNCTIONS_WHITELIST = {}

### Torch functions
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value): value
        for key, value in getmembers(torch._C._VariableFunctions, isbuiltin)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value): value
        for key, value in getmembers(torch.nn.functional, isfunction)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value): value
        for key, value in getmembers(torch._C._nn, isbuiltin)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value, module_name="Tensor"): value
        for key, value in getmembers(
            torch.Tensor, lambda x: ismethoddescriptor(x) or isfunction(x)
        )
    }
)
FUNCTIONS_WHITELIST.update({get_function_name(torch.nn.Parameter): torch.nn.Parameter})

FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value): value
        for key, value in getmembers(torch.optim, isclass)
        if issubclass(value, torch.optim.Optimizer)
    }
)
FUNCTIONS_WHITELIST.update({get_function_name(DataLoader): DataLoader})
### operator functions
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value): value
        for key, value in getmembers(operator, isbuiltin)
        if not key.startswith("_")
    }
)
### einops functions
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value): value
        for key, value in getmembers(einops.einops, isfunction)
    }
)

### nnsight functions
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(bool): bool,
        get_function_name(bytes): bytes,
        get_function_name(int): int,
        get_function_name(float): float,
        get_function_name(str): str,
        get_function_name(complex): complex,
        get_function_name(bytearray): bytearray,
        get_function_name(tuple): tuple,
        get_function_name(list): list,
        get_function_name(set): set,
        get_function_name(dict): dict,
        get_function_name(print): print,
        get_function_name(setattr): setattr,
        get_function_name(util.fetch_attr): util.fetch_attr,
        get_function_name(Proxy.call): Proxy.call,
    }
)

### protocols
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(protocol): protocol
        for key, protocol in getmembers(protocols)
        if isinstance(protocol, type) and issubclass(protocol, protocols.Protocol)
    }
)

FUNCTIONS_WHITELIST.update(
    {
        get_function_name(protocol): protocol
        for key, protocol in getmembers(intervention_protocols)
        if isinstance(protocol, type) and issubclass(protocol, protocols.Protocol)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(protocol): protocol
        for key, protocol in getmembers(contexts)
        if isinstance(protocol, type) and issubclass(protocol, protocols.Protocol)
    }
)

FUNCTIONS_WHITELIST.update(
    {
        get_function_name(protocol): protocol
        for key, protocol in getmembers(intervention_contexts)
        if isinstance(protocol, type) and issubclass(protocol, protocols.Protocol)
    }
)

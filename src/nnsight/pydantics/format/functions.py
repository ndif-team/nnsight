import operator
from inspect import (getmembers, isbuiltin, isfunction, ismethod,
                     ismethoddescriptor)

import einops
import torch

from ... import intervention, util
from ...contexts.accum import Iterator
from ...tracing import protocols
from ...tracing.Proxy import Proxy


def get_function_name(fn, module_name=None):
    if isinstance(fn, str):
        return fn

    if module_name is not None:
        return f"{module_name}.{fn.__name__}"

    module_name = getattr(fn, "__module__", None)

    if module_name is None:
        return fn.__qualname__

    return f"{module_name}.{fn.__qualname__}"


FUNCTIONS_WHITELIST = {}
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
        get_function_name(value, module_name="Tensor"): value
        for key, value in getmembers(
            torch.Tensor, lambda x: ismethoddescriptor(x) or isfunction(x)
        )
    }
)
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
        get_function_name(setattr): setattr,
        get_function_name(util.fetch_attr): util.fetch_attr,
        get_function_name(Proxy.proxy_call): Proxy.proxy_call,
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
        for key, protocol in getmembers(Iterator)
        if isinstance(protocol, type) and issubclass(protocol, protocols.Protocol)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(protocol): protocol
        for key, protocol in getmembers(intervention)
        if isinstance(protocol, type) and issubclass(protocol, protocols.Protocol)
    }
)

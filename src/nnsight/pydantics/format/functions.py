import operator
from inspect import (getmembers, isbuiltin, isfunction, ismethod,
                     ismethoddescriptor)

import einops
import torch

from ... import util
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
        for key, value in getmembers(torch.Tensor, ismethoddescriptor)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value): value
        for key, value in getmembers(operator, isbuiltin)
        if not key.startswith("_")
    }
)
FUNCTIONS_WHITELIST.update(
    {
        get_function_name(value): value
        for key, value in getmembers(einops.einops, isfunction)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        "null": "null",
        "module": "module",
        "argument": "argument",
        "swp": "swp",
        "grad": "grad",
        get_function_name(util.fetch_attr): util.fetch_attr,
        get_function_name(Proxy.proxy_call): Proxy.proxy_call,
    }
)

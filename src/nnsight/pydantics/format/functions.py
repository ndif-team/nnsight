import operator
from inspect import getmembers, isbuiltin, isfunction, ismethoddescriptor

import einops
import torch

from ... import util
from ...tracing.Proxy import Proxy


def get_function_name(fn):
    if isinstance(fn, str):
        return fn

    return f"{getattr(fn, '__module__', '')}.{fn.__qualname__}"


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
        get_function_name(value): value
        for key, value in getmembers(torch._C._TensorBase, ismethoddescriptor)
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

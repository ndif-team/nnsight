import operator
from inspect import getmembers, isbuiltin, ismethoddescriptor

import torch

from ... import util
from ...tracing.Proxy import Proxy

FUNCTIONS_WHITELIST = {}
FUNCTIONS_WHITELIST.update(
    {
        f"_VariableFunctionsClass.{key}": value
        for key, value in getmembers(torch._C._VariableFunctions, isbuiltin)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        f"Tensor.{key}": value
        for key, value in getmembers(torch._C._TensorBase, ismethoddescriptor)
    }
)
FUNCTIONS_WHITELIST.update(
    {
        f"{key}": value
        for key, value in getmembers(operator, isbuiltin)
        if not key.startswith("_")
    }
)
FUNCTIONS_WHITELIST.update(
    {
        "null": "null",
        "module": "module",
        "argument": "argument",
        "swp": "swp",
        "grad": "grad",
        "fetch_attr": util.fetch_attr,
        "Proxy.proxy_call": Proxy.proxy_call,
    }
)

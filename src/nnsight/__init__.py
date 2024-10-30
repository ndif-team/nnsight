# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                       #
#      ::::    ::: ::::    :::  :::::::: ::::::::::: ::::::::  :::    ::: :::::::::::       :::::::       ::::::::      #
#      :+:+:   :+: :+:+:   :+: :+:    :+:    :+:    :+:    :+: :+:    :+:     :+:          :+:   :+:     :+:    :+:     #
#      :+:+:+  +:+ :+:+:+  +:+ +:+           +:+    +:+        +:+    +:+     +:+          +:+  :+:+            +:+     #
#      +#+ +:+ +#+ +#+ +:+ +#+ +#++:++#++    +#+    :#:        +#++:++#++     +#+          +#+ + +:+         +#++:      #
#      +#+  +#+#+# +#+  +#+#+#        +#+    +#+    +#+   +#+# +#+    +#+     +#+          +#+#  +#+            +#+     #
#      #+#   #+#+# #+#   #+#+# #+#    #+#    #+#    #+#    #+# #+#    #+#     #+#          #+#   #+# #+# #+#    #+#     #
#      ###    #### ###    ####  ######## ########### ########  ###    ###     ###           #######  ###  ########      #
#                                                                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import os
from functools import wraps
from typing import Any, Callable, Dict, Union

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nnsight")
except PackageNotFoundError:
    __version__ = "unknown version"

import torch
import yaml

from .util import Patch, Patcher
from .schema.config import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

from .logger import logger, remote_logger
from .intervention import NNsight
from .modeling.language import LanguageModel
# from .tracing.Proxy import proxy_wrapper

logger.disabled = not CONFIG.APP.LOGGING
remote_logger.disabled = not CONFIG.APP.REMOTE_LOGGING

# Below do default patching:
DEFAULT_PATCHER = Patcher()

import math
from inspect import getmembers, isbuiltin, isfunction

# import einops

# for key, value in getmembers(einops.einops, isfunction):
#     DEFAULT_PATCHER.add(Patch(einops.einops, proxy_wrapper(value), key))
# for key, value in getmembers(math, isbuiltin):
#     DEFAULT_PATCHER.add(Patch(math, proxy_wrapper(value), key))

# Tensor creation operations
from torch._subclasses.fake_tensor import FakeTensor


def fake_bool(self):
    return True


DEFAULT_PATCHER.add(Patch(FakeTensor, fake_bool, "__bool__"))


def fake_tensor_new_wrapper(fn):

    @wraps(fn)
    def inner(cls, fake_mode, elem, device, constant=None):

        if isinstance(elem, FakeTensor):

            return elem

        else:

            return fn(cls, fake_mode, elem, device, constant=constant)

    return inner


DEFAULT_PATCHER.add(
    Patch(FakeTensor, fake_tensor_new_wrapper(FakeTensor.__new__), "__new__")
)

DEFAULT_PATCHER.__enter__()

# from .contexts.Context import GlobalTracingContext

# bool = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.bool
# bytes = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.bytes
# int = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.int
# float = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.float
# str = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.str
# complex = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.complex
# bytearray = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.bytearray
# tuple = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.tuple
# list = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.list
# set = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.set
# dict = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.dict
# apply = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply
# log = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.log
# cond = GlobalTracingContext.GLOBAL_TRACING_CONTEXT.cond

# import inspect

# from . import util
# from .intervention.graph import InterventionProxy


# def trace(fn: Callable):
#     """Helper decorator to add a function to the intervention graph via `.apply(...)`.
#     This is opposed to entering the function during tracing and tracing all inner operations.

#     Args:
#         fn (Callable): Function to apply.

#     Returns:
#         Callable: Traceable function.
#     """

#     @wraps(fn)
#     def inner(*args, **kwargs):

#         return apply(fn, *args, **kwargs)

#     return inner


# def local(object: Callable | InterventionProxy):
#     """Helper decorator to add a function to the intervention graph via `.apply(...)`
#     AND convert all input Proxies to local ones via `.local()`.
    
#     If a non-function is passed in, its assumed to be an `InterventionProxy` and `.local()` is called and returned.

#     Args:
#         object ( Callable | InterventionProxy): Function to apply or Proxy to make local.

#     Returns:
#         Callable | InterventionProxy: Traceable local function or local Proxy.
#     """
    
#     if inspect.isroutine(object):

#         fn = trace(object)

#         @wraps(fn)
#         def inner(*args, **kwargs):

#             args, kwargs = util.apply(
#                 (args, kwargs), lambda x: x.local(), InterventionProxy
#             )

#             return fn(*args, **kwargs)

#         return inner
    
#     return object.local()


# def remote(object: Callable | Any):
#     """Helper decorator to add a function to the intervention graph via `.apply(...)`
#     AND convert all input Proxies to downloaded local ones via `.local()`
#     AND convert the output to an uploaded remote one via `remote()`.
    
#     If a non-function is passed in, `remote(object)` is called and returned.

#     Args:
#         object ( Callable | Any): Function to apply or object to make remote.

#     Returns:
#         Callable | InterventionProxy: Traceable local -> remote function or remote Proxy.
#     """

#     if inspect.isroutine(object):

#         fn = local(object)

#         @wraps(fn)
#         def inner(*args, **kwargs):

#             return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.remote(
#                 fn(*args, **kwargs)
#             )

#         return inner

#     return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.remote(object)

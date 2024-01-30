import os

import yaml

from .patching import *
from .pydantics.Config import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

from .logger import logger
from .models.DiffuserModel import DiffuserModel
from .models.LanguageModel import LanguageModel
from .models.NNsightModel import NNsightModel
from .module import Module
from .patching import Patch, Patcher
from .tracing.Proxy import proxy_wrapper

logger.disabled = not CONFIG.APP.LOGGING


# Below do default patching:
DEFAULT_PATCHER = Patcher()

from functools import wraps
from inspect import getmembers, isfunction

import einops
import torch

for key, value in getmembers(einops.einops, isfunction):
    DEFAULT_PATCHER.add(Patch(einops.einops, proxy_wrapper(value), key))


DEFAULT_PATCHER.add(Patch(torch, proxy_wrapper(torch.gather), "gather"))


# Need to patch repeat_interleave to work with meta tensors
# Computes appropriate shape if meta. Otherwise just call repeat_interleave
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
    Patch(
        torch, repeat_interleave_wrapper(torch.repeat_interleave), "repeat_interleave"
    )
)


def noop_wrapper(fn):
    @wraps(fn)
    def noop(input: torch.Tensor, *args, **kwargs):
        if input.device.type == "meta":
            return input

        else:
            return fn(input, *args, **kwargs)

    return noop


DEFAULT_PATCHER.add(Patch(torch.Tensor, noop_wrapper(torch.Tensor.cpu), "cpu"))


def onehot_wrapper(fn):
    @wraps(fn)
    def onehot(input: torch.Tensor, num_classes=-1):
        if input.device.type == "meta":
            return torch.zeros((*input.shape, num_classes), device="meta")

        else:
            return fn(input, num_classes=num_classes)

    return onehot


DEFAULT_PATCHER.add(
    Patch(torch.nn.functional, onehot_wrapper(torch.nn.functional.one_hot), "one_hot")
)


DEFAULT_PATCHER.add(Patch(torch.Tensor, noop_wrapper(torch.Tensor.tolist), "tolist"))


def meta_nonzero(input: torch.Tensor, *args, as_tuple=False, **kwargs):
    output = torch.zeros((input.numel(), input.ndim), device="meta", dtype=torch.long)

    if as_tuple:
        return tuple([output[:, i] for i in range(input.ndim)])

    return output


def meta_nonzero_wrapper(fn):
    @wraps(fn)
    def inner(input: torch.Tensor, *args, **kwargs):
        if input.device.type == "meta":
            return meta_nonzero(input, *args, **kwargs)

        else:
            return fn(input, *args, **kwargs)

    return inner


DEFAULT_PATCHER.add(
    Patch(torch.Tensor, meta_nonzero_wrapper(torch.Tensor.nonzero), "nonzero")
)


def meta_where_wrapper(fn):
    @wraps(fn)
    def where(input: torch.Tensor, *args, **kwargs):
        if input.device.type == "meta":
            if len(args) > 0:
                dtype = (
                    args[0].dtype
                    if isinstance(args[0], torch.Tensor)
                    else type(args[0])
                )
                return torch.zeros_like(torch.broadcast_tensors(input, args[0])[0], dtype=input.dtype, device="meta")
            return meta_nonzero(input, as_tuple=True)

        else:
            return fn(input, *args, **kwargs)

    return where


DEFAULT_PATCHER.add(Patch(torch, meta_where_wrapper(torch.where), "where"))


DEFAULT_PATCHER.__enter__()

from torch._meta_registrations import (_meta_lib_dont_use_me_use_register_meta,
                                       aten, global_decomposition_table,
                                       register_meta)


# Function which "activates" the most recent meta registered function.
def activate_recent_meta():
    op_overload, fn = list(global_decomposition_table["meta"].items())[-1]
    op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)
    _meta_lib_dont_use_me_use_register_meta.impl(op_overload, fn)


# Need to patch local_scalar_dense_meta for meta.
# In non-meta tensors this returns the singular value in tensors with one value
# In meta, lets just return 0
@register_meta(aten._local_scalar_dense)
def local_scalar_dense_meta(A):
    return 0


activate_recent_meta()

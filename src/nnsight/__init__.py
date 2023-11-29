import os

import yaml

from .patching import *
from .pydantics.Config import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

from .models.AbstractModel import AbstractModel
from .models.DiffuserModel import DiffuserModel
from .models.LanguageModel import LanguageModel
from .module import Module
from .patching import Patch, Patcher

# Below do default patching:
DEFAULT_PATCHER = Patcher()

from functools import wraps

import torch


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
    Patch(torch.repeat_interleave, repeat_interleave_wrapper(torch.repeat_interleave))
)


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

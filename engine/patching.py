from __future__ import annotations

import importlib
from typing import List


class Patch:
    def __init__(self, obj, replacement) -> None:
        self.obj = obj
        self.replacement = replacement

    def patch(self):
        module = importlib.import_module(self.obj.__module__)

        setattr(module, self.obj.__name__, self.replacement)

    def restore(self):
        module = importlib.import_module(self.obj.__module__)

        setattr(module, self.obj.__name__, self.obj)


class Patcher:
    def __init__(self, patches: List[Patch] = None) -> None:
        self.patches = patches or []

    def add(self, patch: Patch):
        self.patches.append(patch)

        patch.patch()

    def __enter__(self) -> Patcher:
        for patch in self.patches:
            patch.patch()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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

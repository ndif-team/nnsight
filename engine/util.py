from typing import Union

import torch

Primative = Union[str, int, float, bool]
Value = Union[Primative, torch.Tensor]


def apply(data, fn, cls):

    if isinstance(data, cls):

        return fn(data)

    if isinstance(data, list):

        return [apply(_data, fn, cls) for _data in data]

    if isinstance(data, tuple):

        return tuple([apply(_data, fn, cls) for _data in data])

    if isinstance(data, dict):

        return {key: apply(value, fn, cls) for key, value in data.items()}

    raise ValueError()
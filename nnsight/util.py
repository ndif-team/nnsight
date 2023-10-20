import time
from functools import wraps
from typing import Any, Callable, Type, Union

import torch

Primative = Union[str, int, float, bool]
Value = Union[Primative, torch.Tensor]


def apply(data: Any, fn: Callable, cls: type):
    if isinstance(data, cls):
        return fn(data)

    if isinstance(data, list):
        return [apply(_data, fn, cls) for _data in data]

    if isinstance(data, tuple):
        return tuple([apply(_data, fn, cls) for _data in data])

    if isinstance(data, dict):
        return {key: apply(value, fn, cls) for key, value in data.items()}

    return data


def fetch_attr(object: object, target: str):
    target_atoms = target.split(".")
    for i, atom in enumerate(target_atoms):
        object = getattr(object, atom)
    return object


def wrap(object: object, wrapper: Type, *args, **kwargs):
    if isinstance(object, wrapper):
        return object

    object.__class__ = type(object.__class__.__name__, (wrapper, object.__class__), {})

    wrapper.__init__(object, *args, **kwargs)

    return object


def timed(func, lggr):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        lggr.debug(f"Method `{func.__qualname__}` ran in {round(end - start, 6)}s")
        return result

    return wrapper


def cross_entropy_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    shift: bool = False,
    avg_batch: bool = True,
    avg_token: bool = True,
):
    logits = logits.cpu()
    target_ids = target_ids.cpu()

    if logits.ndim == 2:
        logits = logits.unsqueeze(0)

    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)

    assert logits.ndim == 3
    assert target_ids.ndim == 2
    assert logits.size(0) == target_ids.size(0)
    assert logits.size(1) == target_ids.size(1)

    if shift:
        logits = logits[:, :-1]
        target_ids = target_ids[:, 1:]

    target_ids = target_ids.long()

    batch_losses = []

    for batch_idx in range(len(logits)):
        batch_loss = torch.nn.functional.cross_entropy(
            logits[batch_idx],
            target_ids[batch_idx],
            reduction="mean" if avg_token else "none",
        )
        batch_losses.append(batch_loss)

    batch_losses = torch.stack(batch_losses)

    if avg_batch:
        batch_losses = batch_losses.mean(dim=0)

    return batch_losses


class WrapperModule(torch.nn.Module):
    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args

"""Module for utility functions and classes used throughout the package."""

import time
from functools import wraps
from typing import Any, Callable, Collection, Type

import torch


def apply(data: Collection, fn: Callable, cls: Type) -> Collection:
    """Applies some function to all members of a collection of a give type (or types)

    Args:
        data (Collection): Collection to apply function to.
        fn (Callable): Function to apply.
        cls (type): Type or Types to apply function to.

    Returns:
        Collection: Same kind of collection as data, after then fn has been applied to members of given type.
    """
    if isinstance(data, cls):
        return fn(data)
    
    data_type = type(data)

    if data_type == list:
        return [apply(_data, fn, cls) for _data in data]

    if data_type == tuple:
        return tuple([apply(_data, fn, cls) for _data in data])

    if data_type == dict:
        return {key: apply(value, fn, cls) for key, value in data.items()}

    return data


def fetch_attr(object: object, target: str) -> Any:
    """Retrieves an attribute from an object hierarchy given an attribute path. Levels are separated by '.' e.x (transformer.h.1)

    Args:
        object (object): Root object to get attribute from.
        target (str): Attribute path as '.' separated string.

    Returns:
        Any: Fetched attribute.
    """
    target_atoms = target.split(".")
    for i, atom in enumerate(target_atoms):
        object = getattr(object, atom)
    return object


def wrap(object: object, wrapper: Type, *args, **kwargs) -> object:
    """Wraps some object given some wrapper type.
    Updates the __class__ attribute of the object and calls the wrapper type's __init__ method.

    Args:
        object (object): Object to wrap.
        wrapper (Type): Type to wrap the object in.

    Returns:
        object: Wrapped object.
    """
    if isinstance(object, wrapper):
        return object

    object.__class__ = type(object.__class__.__name__, (wrapper, object.__class__), {})

    wrapper.__init__(object, *args, **kwargs)

    return object


def cross_entropy_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    shift: bool = False,
    avg_batch: bool = True,
    avg_token: bool = True,
) -> torch.Tensor:
    """Helper function for cross entropy loss.

    Args:
        logits (torch.Tensor): Logits tensor of shape (batch size, n tokens, n features) or (n tokens, n features).
        target_ids (torch.Tensor): Target ids tensor of shape (batch size, n tokens) or (n tokens).
        shift (bool, optional): If to ignore the last token of logits and first token of target ids. Defaults to False.
        avg_batch (bool, optional): If to average the loss across batch. Defaults to True.
        avg_token (bool, optional): If to average the loss across tokens. Defaults to True.

    Returns:
        torch.Tensor: Loss.
    """
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
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args


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

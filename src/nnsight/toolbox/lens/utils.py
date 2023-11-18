import torch
from abc import ABC, abstractmethod

from nnsight import LanguageModel, Module
from typing import Any, Callable, Dict, List, Union

from functools import reduce

import torch

def shift_preds(x: torch.Tensor, shift: int):
    """Shift predictions by a given amount.

    Args:
        x: (batch x seq_len) predictions to shift.
        shift: Amount to shift by. Positive values take from the end, negative values
            from the start.

    Returns:
        (batch x (seq_len - shift)) predictions shifted by the given amount.
    """
    if shift > 0:
        return x[:, :-shift]
    if shift < 0:
        return x[:, -shift:]

    return x

# implement kl loss

    
    
    
    
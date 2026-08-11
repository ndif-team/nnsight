"""How many ways a checkpoint's weights can be split, read from its config.

Answered before anything is loaded, because a server placing a model has to
decide how many cards to give it *first*. The whole question is divisibility:
transformers shards attention by head and the MLP by its intermediate
dimension, and its all-gather assumes every rank holds an equal piece, so a
degree that doesn't divide those evenly is not a slower option — it fails.

The answer is a single number, the largest degree that works. Every degree that
works is a divisor of it, so a caller wanting *n* ranks takes the smallest
divisor ``>= n``; if there is none the model has to be spread another way.
"""

from __future__ import annotations

from math import gcd
from typing import Any, Optional

from .interleaver import UNSUPPORTED

#: Config fields a tensor-parallel degree has to divide. Attention is split by
#: head, so both head counts must divide; the MLP is split along its intermediate
#: dimension. A field a config doesn't have is not a constraint — a model with no
#: separate key/value head count shards its attention by ``num_attention_heads``
#: alone.
_DIVIDES: tuple[str, ...] = (
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
)


def _text_config(config: Any) -> Any:
    """The config carrying the transformer dimensions.

    A multimodal checkpoint keeps them on a nested ``text_config`` while the
    outer config holds the composition, so reading the outer one finds nothing
    to divide and would call every degree workable.
    """
    return getattr(config, "text_config", None) or config


def max_tp_size(config: Any) -> Optional[int]:
    """The largest tensor-parallel degree ``config``'s model supports.

    ``None`` when it cannot be split at all: no plan to shard by, or a plan that
    shards in a way interventions can't be shown whole (see
    [`UNSUPPORTED`][nnsight.modeling.tp.interleaver.UNSUPPORTED] — the
    expert-parallel family, which slices by expert). Refusing here keeps a model
    that would fail at load from being *placed* as though it could be split.
    """
    plan = getattr(config, "base_model_tp_plan", None)
    if not plan:
        return None

    if any(style in UNSUPPORTED for style in plan.values()):
        return None

    dimensions = [
        value
        for name in _DIVIDES
        if isinstance(value := getattr(_text_config(config), name, None), int) and value
    ]
    if not dimensions:
        return None

    # Every degree that divides all of them divides their gcd, and vice versa,
    # so the gcd *is* the largest workable degree and its divisors are the rest.
    limit = gcd(*dimensions) if len(dimensions) > 1 else dimensions[0]
    return limit if limit > 1 else None

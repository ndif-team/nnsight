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

from .fragments import SHARDED_SIDES

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

    ``None`` when it cannot be split at all: no plan to shard by, or a plan
    containing a style [`TPFragments.instrument`][nnsight.modeling.tp.fragments.TPFragments.instrument]
    will refuse. Refusing here keeps a model that would fail at load from being
    *placed* as though it could be split.

    The two must refuse **the same set**, which is why this asks
    [`SHARDED_SIDES`][nnsight.modeling.tp.fragments.SHARDED_SIDES] rather than
    only [`UNSUPPORTED`][nnsight.modeling.tp.fragments.UNSUPPORTED]. A style in
    neither — one transformers added, or one it never registered in
    ``ALL_PARALLEL_STYLES`` — used to pass here and raise there, so a server
    would allocate the cards, load the weights across them, and only then find
    out. ``Llama4Config`` does exactly this: its plan is ``colwise_rep``, which
    is in no list and no registry.
    """
    plan = getattr(config, "base_model_tp_plan", None)
    if not plan:
        return None

    if any(style not in SHARDED_SIDES for style in plan.values()):
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


class UnshardableCheckpoint(ValueError):
    """A tensor-parallel degree was asked for that this checkpoint cannot serve."""


def requested_tp_size(distributed_config: Any) -> Optional[int]:
    """The degree a ``distributed_config`` asks for, or ``None`` if it asks for none.

    Accepts the dataclass or a plain dict, because transformers does.
    """
    if distributed_config is None:
        return None

    if isinstance(distributed_config, dict):
        size = distributed_config.get("tp_size")
    else:
        size = getattr(distributed_config, "tp_size", None)

    return size if isinstance(size, int) and size > 1 else None


def check_tp_request(config: Any, tp_size: Optional[int]) -> None:
    """Raise unless ``tp_size`` is a degree ``config``'s model can really be split into.

    transformers does not check this. Asked to shard a checkpoint with no plan it
    shards *nothing*: ``verify_tp_plan`` returns early on a ``None`` plan and
    ``apply_tensor_parallelism`` installs no hooks, so every rank quietly loads a
    complete copy of the weights. Nothing errors and nothing warns — the model
    answers correctly off one rank while the other cards hold redundant copies,
    so the only symptom is *n* times the memory for one model's worth of work.

    That is worth refusing rather than reporting, because there is no reading of
    "shard this over 4 GPUs" that is served by putting the whole thing on each of
    them. Raising here also puts the failure before the weights are fetched,
    where the message can still say what to do about it.

    Raises:
        UnshardableCheckpoint: if the model cannot be split at all, or not into
            exactly ``tp_size`` pieces.
    """
    if tp_size is None:
        return

    limit = max_tp_size(config)
    if limit is None:
        raise UnshardableCheckpoint(
            f"this checkpoint cannot be split tensor-parallel, so tp_size={tp_size} "
            "would load a whole copy of it onto every rank rather than a shard. "
            "Either it publishes no `base_model_tp_plan`, its plan uses a style "
            "nnsight cannot gather, or its dimensions divide no degree above 1. "
            "Load it without `distributed_config` — on one GPU, or spread over "
            "several with `device_map`."
        )

    if limit % tp_size:
        workable = sorted(size for size in range(2, limit + 1) if limit % size == 0)
        raise UnshardableCheckpoint(
            f"this checkpoint splits at most {limit} ways and not into {tp_size} "
            f"pieces: a degree has to divide the dimensions evenly, so the ones "
            f"that work are {workable}. transformers would shard what it could and "
            "gather as though every rank held an equal piece, which does not fail "
            "so much as return the wrong shape."
        )

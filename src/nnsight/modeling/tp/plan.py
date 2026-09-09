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

from .fragments import SIDES

#: What an *expert*-parallel degree has to divide instead. Expert parallelism
#: distributes whole experts rather than slicing a tensor axis, so the head
#: counts and the intermediate size are irrelevant to it and the expert count is
#: the only constraint. transformers raises on an uneven split itself
#: (``EpRouterParallel`` checks ``num_experts % ep_size``), but by then the
#: weights are already being read.
_EXPERT_DIVIDES: tuple[str, ...] = ("num_local_experts", "num_experts")

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


def max_tp_size(config: Any, expert_parallel: bool = False) -> Optional[int]:
    """The largest tensor-parallel degree ``config``'s model supports.

    ``None`` when it cannot be split at all: no plan to shard by, or a plan
    containing a style [`TPFragments.instrument`][nnsight.modeling.tp.fragments.TPFragments.instrument]
    will refuse. Refusing here keeps a model that would fail at load from being
    *placed* as though it could be split.

    The two must refuse **the same set**, which is why this asks
    [`SIDES`][nnsight.modeling.tp.fragments.SIDES] rather than
    only [`UNSUPPORTED`][nnsight.modeling.tp.fragments.UNSUPPORTED]. A style in
    neither — one transformers added, or one it never registered in
    ``ALL_PARALLEL_STYLES`` — used to pass here and raise there, so a server
    would allocate the cards, load the weights across them, and only then find
    out. ``Llama4Config`` does exactly this: its plan is ``colwise_rep``, which
    is in no list and no registry.
    """
    # Expert parallelism is answered from the model's *expert* plan, which is what
    # transformers applies for it; a checkpoint can publish one and no
    # tensor-parallel plan at all (gpt-oss does), and reading the wrong field
    # would call it unshardable.
    fields = _EXPERT_DIVIDES if expert_parallel else _DIVIDES
    plan = getattr(
        config, "base_model_ep_plan" if expert_parallel else "base_model_tp_plan", None
    )
    if not plan:
        return None

    if any(style not in SIDES for style in plan.values()):
        return None

    dimensions = [
        value
        for name in fields
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


def requested_expert_parallel(distributed_config: Any) -> bool:
    """Whether this ``distributed_config`` asks for expert parallelism.

    Accepts the dataclass or a plain dict, because transformers does.
    """
    if distributed_config is None:
        return False
    if isinstance(distributed_config, dict):
        return bool(distributed_config.get("enable_expert_parallel"))
    return bool(getattr(distributed_config, "enable_expert_parallel", False))


def check_tp_request(
    config: Any, tp_size: Optional[int], expert_parallel: bool = False
) -> None:
    """Raise unless ``tp_size`` is a degree ``config``'s model can really be split into.

    transformers does not check this, and its two failure shapes are both worth
    refusing. Asked to shard a checkpoint with *no plan* it shards nothing:
    ``verify_tp_plan`` returns early on a ``None`` plan and
    ``apply_tensor_parallelism`` installs no hooks, so every rank quietly loads a
    complete copy of the weights — nothing errors, nothing warns, and the only
    symptom is *n* times the memory for one model's worth of work. Asked for a
    degree the plan *cannot divide* (SmolLM2's 9 heads at 2 ranks), it loads the
    checkpoint sharded anyway — DTensor splits the 576 q_proj columns evenly,
    4.5 heads per rank — and the first forward dies on
    ``RuntimeError: shape '[1, 9, -1, 64]' is invalid for input of size 2592``,
    a reshape of the local tensor by the global head count, naming nothing about
    tensor parallelism. Silent waste or a late opaque crash; the refusal here is
    early and says what to do.

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

    limit = max_tp_size(config, expert_parallel)
    if limit is None:
        axis = "expert-parallel" if expert_parallel else "tensor-parallel"
        field = "base_model_ep_plan" if expert_parallel else "base_model_tp_plan"
        divides = "its expert count divides" if expert_parallel else "its dimensions divide"
        raise UnshardableCheckpoint(
            f"this checkpoint cannot be split {axis}, so tp_size={tp_size} "
            "would load a whole copy of it onto every rank rather than a shard. "
            f"Either it publishes no `{field}`, its plan uses a style "
            f"nnsight cannot gather, or {divides} no degree above 1. "
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

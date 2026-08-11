"""Tracing a model sharded with transformers tensor parallelism.

Nothing here needs installing or enabling. A
[`HuggingFaceModel`][nnsight.modeling.huggingface.HuggingFaceModel] is always
built with a [`TPInterleaver`][nnsight.modeling.tp.interleaver.TPInterleaver],
which stays inert unless it finds the model actually sharded. See
[`interleaver`][nnsight.modeling.tp.interleaver] for how a shard is turned back
into the whole tensor a user asked for.
"""

from .interleaver import (
    MINIMUM_TRANSFORMERS,
    SHARDED_SIDES,
    UNSUPPORTED,
    TPInterleaver,
    UnsupportedParallelStyle,
    UnsupportedTransformersVersion,
    is_sharded,
)
from .plan import max_tp_size

__all__ = [
    "MINIMUM_TRANSFORMERS",
    "SHARDED_SIDES",
    "UNSUPPORTED",
    "TPInterleaver",
    "UnsupportedParallelStyle",
    "UnsupportedTransformersVersion",
    "is_sharded",
    "max_tp_size",
]

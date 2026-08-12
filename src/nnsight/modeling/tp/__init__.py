"""Tracing a model sharded with transformers tensor parallelism.

Nothing here needs installing or enabling. A
[`HuggingFaceModel`][nnsight.modeling.huggingface.HuggingFaceModel] is always
built with a [`TPFragments`][nnsight.modeling.tp.fragments.TPFragments], which
stays inert unless it finds the model actually sharded. See
[`fragments`][nnsight.modeling.tp.fragments] for which values are pieces and how
they are reassembled, and
[`nnsight.intervention.fragments`][nnsight.intervention.fragments] for when.
"""

from .fragments import (
    MINIMUM_TRANSFORMERS,
    SHARDED_SIDES,
    UNSUPPORTED,
    TPFragments,
    UnsupportedParallelStyle,
    UnsupportedTransformersVersion,
    is_sharded,
)
from .plan import (
    UnshardableCheckpoint,
    check_tp_request,
    max_tp_size,
    requested_tp_size,
)

__all__ = [
    "MINIMUM_TRANSFORMERS",
    "SHARDED_SIDES",
    "UNSUPPORTED",
    "TPFragments",
    "UnsupportedParallelStyle",
    "UnsupportedTransformersVersion",
    "is_sharded",
    "UnshardableCheckpoint",
    "check_tp_request",
    "max_tp_size",
    "requested_tp_size",
]

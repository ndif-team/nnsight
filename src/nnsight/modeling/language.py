"""A causal-language-model name over [`TransformersModel`][nnsight.modeling.transformers.TransformersModel] (deprecated).

[`TransformersModel`][nnsight.modeling.transformers.TransformersModel] already generates through the model and returns token
ids (see its [`generate`][nnsight.modeling.transformers.TransformersModel.generate]),
so [`LanguageModel`][nnsight.modeling.language.LanguageModel] adds nothing of its own — it just pins the
text-generation task and takes ``tokenizer_kwargs`` at load. Prefer
``TransformersModel(repo_id, task="text-generation")`` directly; this name is kept
for backwards compatibility and warns on construction.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from .. import NNsightDeprecationWarning
from .transformers import TransformersModel


class LanguageModel(TransformersModel):
    """Deprecated: a [`TransformersModel`][nnsight.modeling.transformers.TransformersModel]
    pinned to the text-generation task.

    Use ``TransformersModel(repo_id, task="text-generation")`` instead. Everything
    this class offered — ``generate`` returning token ids, ``pipe`` running the
    pipeline, ``trace``, ``scan`` — lives on [`TransformersModel`][nnsight.modeling.transformers.TransformersModel] now; the
    only extra here is ``tokenizer_kwargs`` (apply the same settings to
    ``model.tokenizer`` yourself, ``padding_side`` being the usual one).
    """

    def __init__(
        self, *args: Any, tokenizer_kwargs: Optional[dict] = None, **kwargs: Any
    ) -> None:
        if type(self) is LanguageModel:
            # Only for the concrete class — a subclass (VisionLanguageModel) warns
            # about itself, not about the LanguageModel it happens to extend.
            warnings.warn(
                "LanguageModel is deprecated; use "
                "TransformersModel(repo_id, task='text-generation') instead.",
                NNsightDeprecationWarning,
                stacklevel=2,
            )
        # A causal LM: use the text-generation pipeline (it is what the checkpoint
        # would usually infer anyway, but be explicit for the LanguageModel name).
        kwargs.setdefault("task", "text-generation")
        super().__init__(*args, **kwargs)
        # The pipeline loads the tokenizer; apply any requested settings to it
        # (padding_side is the common one).
        for key, value in (tokenizer_kwargs or {}).items():
            setattr(self.tokenizer, key, value)

    def _remoteable_class(self) -> type:
        # A deprecated alias for TransformersModel: share its remote key (and so
        # VisionLanguageModel's, which inherits this) so a model deployed as a
        # TransformersModel is reachable when wrapped as either of them.
        return TransformersModel

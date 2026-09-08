"""A vision-language name over [`TransformersModel`][nnsight.modeling.transformers.TransformersModel] (deprecated).

[`TransformersModel`][nnsight.modeling.transformers.TransformersModel] generates through the
model and returns token ids. A vision-language model is the same story with images
alongside the text, so prefer ``TransformersModel(repo_id,
task="image-text-to-text")`` directly. [`VisionLanguageModel`][nnsight.modeling.vlm.VisionLanguageModel] is kept for
backwards compatibility — all it does is pin the task and warn on construction;
input handling (a prompt with ``images=``, or a processor encoding you built
yourself) is the base class's.

.. code-block:: python

    from nnsight import VisionLanguageModel

    model = VisionLanguageModel("llava-hf/llava-1.5-7b-hf", dispatch=True)
    with model.generate(text=prompt, images=[img], max_new_tokens=20) as tracer:
        ids = tracer.result.save()
    print(model.tokenizer.batch_decode(ids))

The image inputs go by keyword (``images=``, ``text=``) — what the task's
processor takes.
"""

from __future__ import annotations

import warnings
from typing import Any

from .. import NNsightDeprecationWarning
from .language import LanguageModel


class VisionLanguageModel(LanguageModel):
    """Deprecated: a [`TransformersModel`][nnsight.modeling.transformers.TransformersModel]
    pinned to the image-text-to-text task.

    Use ``TransformersModel(repo_id, task="image-text-to-text")`` instead. Its
    ``generate`` returns the generated **token ids** from a prompt plus images
    given by keyword (``text=``, ``images=``), the task's processor turning them
    into the model inputs a forward takes; everything else — ``trace``, ``scan`` —
    is inherited.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "VisionLanguageModel is deprecated; use "
            "TransformersModel(repo_id, task='image-text-to-text') instead.",
            NNsightDeprecationWarning,
            stacklevel=2,
        )
        # A vision-language model: use the image-text-to-text pipeline (set before
        # super's setdefault so it wins over LanguageModel's text-generation task).
        kwargs.setdefault("task", "image-text-to-text")
        super().__init__(*args, **kwargs)

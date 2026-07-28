"""A vision-language name over [`TransformersModel`][nnsight.modeling.transformers.TransformersModel] (deprecated).

[`TransformersModel`][nnsight.modeling.transformers.TransformersModel] generates through the
model and returns token ids. A vision-language model is the same story with images
alongside the text, so prefer ``TransformersModel(repo_id,
task="image-text-to-text")`` directly. [`VisionLanguageModel`][nnsight.modeling.vlm.VisionLanguageModel] is kept for
backwards compatibility — it pins the task and runs the processor over the prompt
*and* images before the model's own ``generate`` — and warns on construction.

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
            DeprecationWarning,
            stacklevel=2,
        )
        # A vision-language model: use the image-text-to-text pipeline (set before
        # super's setdefault so it wins over LanguageModel's text-generation task).
        kwargs.setdefault("task", "image-text-to-text")
        super().__init__(*args, **kwargs)

    def _batch_generate(self, invokes: list) -> tuple:
        """Assemble a prompt and its images into model inputs, via the processor.

        Generating goes through the model, which takes ``input_ids``/``pixel_values``
        — not the raw prompt and images — so run the processor here (a pipeline would
        do it internally, but the model can't). The prompt is positional or ``text=``,
        the images ``images=``; anything else in the invoke is a generate kwarg.

        An invoke that already carries a built encoding (no ``text``/``images`` to
        process) is assembled like a forward's input instead.
        """
        if len(invokes) > 1:
            raise NotImplementedError(
                "Batching multimodal generate inputs isn't supported; use one invoke."
            )
        (inputs, kwargs), = invokes
        kwargs = dict(kwargs)
        text = inputs[0] if inputs else kwargs.pop("text", None)
        images = kwargs.pop("images", None)
        if text is None and images is None:
            return self._batch_forward(invokes)  # a prebuilt encoding
        encoding = self.processor(text=text, images=images, return_tensors="pt")
        return tuple(), {**dict(encoding), **kwargs}

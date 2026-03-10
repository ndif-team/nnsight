from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
)

from .language import LanguageModel


class VisionLanguageModel(LanguageModel):
    """NNsight wrapper for vision-language models (VLMs) such as LLaVA, Qwen2-VL, etc.

    Extends :class:`LanguageModel` with an ``AutoProcessor`` that handles
    both text tokenization and image preprocessing.  Text-only inputs
    fall back to the standard :class:`LanguageModel` tokenization path.

    Inputs can include an ``images`` keyword argument containing a single
    PIL image or a list of PIL images.

    Example::

        from nnsight import VisionLanguageModel
        from PIL import Image

        model = VisionLanguageModel(
            "llava-hf/llava-interleave-qwen-0.5b-hf",
            device_map="auto",
            dispatch=True,
        )
        img = Image.open("photo.jpg")

        with model.trace("Describe this image", images=[img]):
            hidden = model.model.layers[-1].output[0].save()

    Args:
        *args: Forwarded to :class:`LanguageModel`.
        processor (Optional[AutoProcessor]): A pre-loaded processor.
            If ``None``, one is loaded from the repo automatically.
        automodel (Type): The ``AutoModel`` class used for loading.
            Defaults to ``AutoModelForImageTextToText``.
        **kwargs: Forwarded to :class:`LanguageModel` and ultimately
            to ``from_pretrained``.

    Attributes:
        processor (AutoProcessor): Handles both text and image inputs.
    """

    def __init__(
        self,
        *args,
        processor: Optional[AutoProcessor] = None,
        automodel: Type = AutoModelForImageTextToText,
        **kwargs,
    ) -> None:

        self.processor: Optional[AutoProcessor] = processor

        super().__init__(*args, automodel=automodel, **kwargs)

    def _load_meta(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        processor_kwargs: Optional[Dict[str, Any]] = {},
        tokenizer_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):
        self._load_processor(repo_id, revision=revision, **processor_kwargs)

        return super()._load_meta(
            repo_id,
            revision=revision,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

    def _load(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        processor_kwargs: Optional[Dict[str, Any]] = {},
        tokenizer_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):
        self._load_processor(repo_id, revision=revision, **processor_kwargs)

        return super()._load(
            repo_id,
            revision=revision,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

    def _load_processor(self, repo_id: str, **kwargs):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(repo_id, **kwargs)

        if self.tokenizer is None:
            self.tokenizer = self.processor.tokenizer
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_tokenizer(self, repo_id: str, **kwargs):
        # If we already have a processor, the tokenizer comes from it.
        # Only fall back to the parent tokenizer loading if no processor is set.
        if self.processor is not None and self.tokenizer is None:
            self.tokenizer = self.processor.tokenizer
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.tokenizer is None:
            super()._load_tokenizer(repo_id, **kwargs)

    # Processor kwargs that should be separated from model kwargs
    _PROCESSOR_KWARGS = {
        "images",
        "image_sizes",
        "do_rescale",
        "do_resize",
        "do_normalize",
        "size",
        "crop_size",
    }

    def _prepare_input(
        self,
        *inputs: Union[
            str,
            List[str],
            List[List[str]],
            List[int],
            List[List[int]],
            torch.Tensor,
            List[torch.Tensor],
            Dict[str, Any],
            BatchEncoding,
        ],
        input_ids: Union[
            List[int], List[List[int]], torch.Tensor, List[torch.Tensor]
        ] = None,
        labels: Any = None,
        attention_mask: Any = None,
        images: Any = None,
        **kwargs,
    ) -> tuple[tuple[Any], dict[str, Any], int]:
        """Prepare text+image inputs for the model.

        When ``images`` is provided, uses the processor to handle both
        text tokenization and image preprocessing.  When no images are
        given, falls back to the standard :class:`LanguageModel` path.
        """

        # If no images, delegate entirely to the parent LanguageModel
        if images is None:
            return super()._prepare_input(
                *inputs,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                **kwargs,
            )

        # Separate processor-specific kwargs from model kwargs
        processor_kwargs = {}
        model_kwargs = {}
        for k, v in kwargs.items():
            if k in self._PROCESSOR_KWARGS or k in self._TOKENIZER_KWARGS:
                processor_kwargs[k] = v
            else:
                model_kwargs[k] = v

        if input_ids is not None:
            assert len(inputs) == 0
            inputs = (input_ids,)

        if len(inputs) == 0:
            return tuple(), model_kwargs, 0

        assert len(inputs) == 1
        inp = inputs[0]

        # If already preprocessed, pass through
        if isinstance(inp, (dict, BatchEncoding)):
            processed = BatchEncoding(inp) if isinstance(inp, dict) else inp
        else:
            # Ensure text is a list for the processor
            if isinstance(inp, str):
                text = [inp]
            elif isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], str):
                text = inp
            else:
                # Token IDs or other formats - use parent path with images handled separately
                return super()._prepare_input(
                    *inputs,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            # Ensure images is a list
            if not isinstance(images, (list, tuple)):
                images = [images]

            processed = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                padding=True,
                **processor_kwargs,
            )

        if attention_mask is not None:
            processed["attention_mask"] = attention_mask

        if labels is not None:
            if isinstance(labels, str) or (
                isinstance(labels, list) and isinstance(labels[0], str)
            ):
                labels = self._tokenize(labels)["input_ids"]
            processed["labels"] = labels

        batch_size = len(processed["input_ids"])

        return (
            tuple(),
            {**processed, **model_kwargs},
            batch_size,
        )

    def _batch(
        self,
        batched_inputs: Optional[Tuple[Tuple[BatchEncoding], Dict[str, Any]]],
        **prepared_kwargs,
    ) -> tuple[tuple[Any], dict[str, Any]]:
        """Combine multiple invokes into a single batch.

        Re-pads input_ids/attention_mask using the tokenizer (same as
        LanguageModel), then concatenates pixel_values and any other
        image-related tensors along the batch dimension.
        """

        batched_inputs = batched_inputs[1]

        if "input_ids" not in batched_inputs:
            return tuple(), {**prepared_kwargs, **batched_inputs}

        # ---- Re-pad input_ids (same as LanguageModel) ----
        batched_labels = batched_inputs.get("labels", None)
        attention_mask = batched_inputs.get("attention_mask", None)

        batched_ids = [
            {"input_ids": ids}
            for ids in [
                *batched_inputs["input_ids"].tolist(),
                *prepared_kwargs["input_ids"].tolist(),
            ]
        ]
        new_batched_inputs = self.tokenizer.pad(batched_ids, return_tensors="pt")

        labels = prepared_kwargs.get("labels", None)
        if labels is not None:
            batched_labels = (
                torch.cat((batched_labels, labels))
                if batched_labels is not None
                else labels
            )

        new_attention_mask = prepared_kwargs.get("attention_mask", None)
        n_old = attention_mask.shape[0] if attention_mask is not None else 0
        left = self.tokenizer.padding_side == "left"

        combined_mask = torch.zeros_like(new_batched_inputs["attention_mask"])

        for row_start, mask in [(0, attention_mask), (n_old, new_attention_mask)]:
            if mask is not None:
                if left:
                    combined_mask[
                        row_start : row_start + mask.shape[0], -mask.shape[1] :
                    ] = mask
                else:
                    combined_mask[
                        row_start : row_start + mask.shape[0], : mask.shape[1]
                    ] = mask

        new_batched_inputs["attention_mask"] = combined_mask

        # ---- Handle image tensors (pixel_values, image_sizes, etc.) ----
        # Collect all keys that are not text-related
        _text_keys = {"input_ids", "attention_mask", "labels", "token_type_ids"}

        # Remove already-handled keys from batched_inputs
        remaining_batched = {
            k: v for k, v in batched_inputs.items() if k not in _text_keys
        }
        remaining_prepared = {
            k: v for k, v in prepared_kwargs.items() if k not in _text_keys
        }

        # Concatenate image-related tensors
        merged_extra = {}
        all_extra_keys = set(remaining_batched.keys()) | set(remaining_prepared.keys())
        for key in all_extra_keys:
            b_val = remaining_batched.get(key)
            p_val = remaining_prepared.get(key)
            if b_val is not None and p_val is not None:
                if isinstance(b_val, torch.Tensor) and isinstance(p_val, torch.Tensor):
                    merged_extra[key] = torch.cat((b_val, p_val), dim=0)
                elif isinstance(b_val, list) and isinstance(p_val, list):
                    merged_extra[key] = b_val + p_val
                else:
                    merged_extra[key] = p_val
            elif b_val is not None:
                merged_extra[key] = b_val
            elif p_val is not None:
                merged_extra[key] = p_val

        return (
            tuple(),
            {**new_batched_inputs, **merged_extra, "labels": batched_labels},
        )

    def _remoteable_model_key(self) -> str:
        return super()._remoteable_model_key()

    def _remoteable_persistent_objects(self) -> dict:
        persistent_objects = super()._remoteable_persistent_objects()
        persistent_objects["Processor"] = self.processor
        return persistent_objects

    def __getstate__(self):
        state = super().__getstate__()
        self.processor._persistent_id = "Processor"
        state["processor"] = self.processor
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.processor = state["processor"]

"""Turning what a trace is given into what a transformers model takes.

An invoke on a [`TransformersModel`][nnsight.modeling.transformers.TransformersModel]
is text, chat messages, an image, token ids, or an encoding, written
positionally or by keyword. `batch_forward` takes every invoke of a trace to
the one set of model inputs its forward runs on:

* `preprocess_invoke` routes one invoke: through the task's processor or its
  ``Pipeline.preprocess`` for text and media, through `encode_pretokenized`
  for token ids and encodings (merged into one by `as_encoding`), or straight
  to the model for an input that is not known to be per-row.
* `collate` pads the rows of all invokes into one batch. The text fields are
  named, with their padding, in `ROW_FIELDS`.
* `supply_position_ids` keeps a left-padded row's tokens at their own positions.

`batch_size` counts the rows an invoke contributes, with the same splitting
rule (`split`) the encoding itself uses. Functions that need the model's
tokenizer, pipeline or task take the model first.
"""

from __future__ import annotations

import numbers
import sys
from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch._guards import detect_fake_mode


# The per-row model inputs of a text forward, and what a row is padded with
# when another in the batch is wider (``None``: the tokenizer's pad token).
# These are split per row, padded and batched; an invoke's other keywords are
# forward arguments, handed over as they are. An encoding carrying a tensor
# outside this table (pixel_values, input_features, cache_position, ...) is
# not known to be per-row, so it is passed through whole, on its own.
ROW_FIELDS = {
    "input_ids": None,
    "inputs_embeds": 0.0,
    "attention_mask": 0,
    "token_type_ids": 0,
    "position_ids": 0,
    "special_tokens_mask": 1,
    "global_attention_mask": 0,
    "labels": -100,
    "start_positions": 0,
    "end_positions": 0,
    "next_sentence_label": 0,
    "decoder_input_ids": None,
    "decoder_inputs_embeds": 0.0,
    "decoder_attention_mask": 0,
    "decoder_position_ids": 0,
}


# The fields with a length of their own rather than the input's. They are
# padded on the right whichever side the input is padded on; ``labels`` joins
# them on an encoder-decoder, where it is the decoder's target.
TARGET_FIELDS = frozenset(
    {
        "decoder_input_ids",
        "decoder_inputs_embeds",
        "decoder_attention_mask",
        "decoder_position_ids",
    }
)


# The fields one row of which can stand for every row of a batch, as the
# models themselves broadcast them.
SHARED_FIELDS = frozenset({"token_type_ids", "position_ids"})


# Non-text arguments a task's *processor* takes. An invoke naming any of these is
# written in processor terms (``trace(prompt, images=[img])``) rather than model
# terms (``trace(input_ids=...)``), so the processor has to run over it first.
PROCESSOR_MEDIA_KEYS = frozenset(
    {"images", "image", "audio", "audios", "videos", "video"}
)


# ``text`` rides along with the media keys but never triggers the processor path on
# its own — text-only input is handled by the ordinary tokenization route.
PROCESSOR_KEYS = PROCESSOR_MEDIA_KEYS | {"text"}


# Tasks a trace refuses, with what to do instead. ``mask-generation``'s
# preprocess runs the model, and ``keypoint-matching``'s unit input is a pair
# of images, which the list convention (one prompt per element) would split.
UNTRACEABLE_TASKS = {
    "mask-generation": (
        "task='mask-generation' has no forward to trace from an image: its "
        "preprocess embeds the image by running the model, then yields one "
        "input per batch of candidate points. Run the whole task with "
        "model.pipe(image), or trace one forward on an encoding you build "
        "yourself: model.image_processor(image, return_tensors='pt'), with "
        "the points you want as input_points=."
    ),
    "keypoint-matching": (
        "task='keypoint-matching' takes a pair of images as one input, "
        "which a trace's list convention (one prompt per element) "
        "would split. Run the whole task with model.pipe([image_a, "
        "image_b]), or trace one forward on an encoding you build "
        "yourself: model.image_processor(images=[image_a, image_b], "
        "return_tensors='pt')."
    ),
}


def batch_size(inputs: tuple, kwargs: dict) -> int:
    """Number of batch rows an invoke's input contributes.

    Model inputs (token ids, an encoding, keywords) count the rows `split`
    makes of their input; a string is one row, a chat conversation one, and a
    list of prompts or images one per entry. Multimodal data passed by keyword
    (``text=``/``images=`` to a VLM's generate) counts as one row. Zero rows
    means params only (e.g. ``max_new_tokens=``), so the trace expects
    ``invoke()`` blocks for the data.
    """
    data = inputs[0] if inputs else None
    encoding = as_encoding(data, kwargs)
    key = input_key(encoding) if encoding is not None else None
    if key is not None:
        return len(split(key, encoding[key])[0])
    if data is None:
        return int(
            any(
                kwargs.get(key) is not None
                for key in ("text", "images", "pixel_values", "input_features")
            )
        )
    if isinstance(data, torch.Tensor):
        return 1 if data.ndim <= 1 else data.shape[0]
    chats = as_chats(data)
    if chats is not None:
        return len(chats)  # a chat conversation is one row, not one per message
    if isinstance(data, (list, tuple)):
        return len(data)
    return 1


def as_chats(data: Any, chat_cls: Optional[type] = None) -> Optional[list]:
    """Chat message(s) -> a list of ``Chat`` inputs (one per conversation).

    Mirrors the chat detection `Pipeline.__call__` does before preprocess,
    which calling `Pipeline.preprocess` directly would otherwise skip.
    Returns ``None`` when ``data`` isn't chat messages. ``chat_cls`` is the
    wrapper class to use — the pipeline's own when it defines one (see
    `chat_cls`); the base ``Chat`` otherwise.
    """
    try:
        from transformers.pipelines.base import Chat, is_valid_message
    except ImportError:
        # transformers before the chat-pipeline refactor has neither
        # helper: treat everything as not-chat and let the pipeline's own
        # input handling take it from here.
        return None

    chat_cls = chat_cls or Chat
    if not isinstance(data, (list, tuple)) or not data:
        return None
    if is_valid_message(data[0]):
        return [chat_cls(list(data))]
    if all(
        isinstance(chat, (list, tuple)) and chat and is_valid_message(chat[0])
        for chat in data
    ):
        return [chat_cls(list(chat)) for chat in data]
    return None


def chat_cls(model) -> Optional[type]:
    """The ``Chat`` wrapper class this model's pipeline expects.

    Most chat pipelines isinstance-check ``transformers.pipelines.base.Chat``
    (or import it as a module attribute, which resolves to the same object),
    but ``any-to-any`` defines its *own* ``Chat`` and checks against that —
    a base-``Chat`` instance falls through to its raw-dict branch and fails
    on ``Chat.copy``. So a pipeline module that carries a ``Chat`` gets its
    own class.
    """
    module = sys.modules.get(type(model.pipeline).__module__)
    return getattr(module, "Chat", None)


def batch_pipe(invokes: list) -> tuple:
    """Hand text prompts to the pipeline, batched with ``batch_size``.

    A single non-text payload (a VLM's ``text=``/``images=`` keywords) is handed
    to the pipeline as-is; only string prompts are combined into a batch.
    """
    prompts: list = []
    forward: dict = {}
    passthrough = None
    for inputs, kwargs in invokes:
        data = inputs[0] if inputs else None
        if isinstance(data, str):
            prompts.append(data)
        elif isinstance(data, (list, tuple)) and data and isinstance(data[0], str):
            prompts.extend(data)
        else:
            passthrough = (inputs, kwargs)
            continue
        forward.update(kwargs)

    if passthrough is not None:
        if len(invokes) > 1:
            raise NotImplementedError(
                "Batching multimodal generate inputs isn't supported; pass a "
                "single text/images payload."
            )
        return passthrough

    # A single prompt keeps the pipeline's scalar-input output shape; a real
    # batch goes as a list with batch_size so it runs as one batch.
    if len(prompts) == 1:
        return (prompts[0],), forward
    return (prompts,), {**forward, "batch_size": len(prompts)}


def batch_forward(model, invokes: list) -> tuple:
    """Preprocess each invoke and pad the results into one forward input.

    Runs on CPU; interleave() moves inputs to the model's device after the
    (possibly lazy) dispatch, so don't touch device here.
    """
    items: list = []
    forward: dict = {}
    for inputs, kwargs in invokes:
        rows, forward_kwargs = preprocess_invoke(model, inputs[0] if inputs else None, kwargs)
        if rows is None:
            if len(invokes) > 1:
                raise NotImplementedError(
                    "Can't batch these inputs; pass text or token ids."
                )
            return forward_kwargs
        # A chunked task decides its own row count, and the batcher counted
        # this invoke's input as its own rows before preprocessing — so with
        # another invoke in the batch every group after this one names rows
        # that belong to someone else, and each invoke's reads and edits land
        # on the wrong ones. Silent, so refuse it.
        if len(invokes) > 1 and len(rows) != model._batch_size(*inputs, **kwargs):
            raise NotImplementedError(
                f"task={model.task!r} splits this invoke into {len(rows)} forward "
                "rows, and a batched trace gives an invoke the rows its input "
                "has — the other invokes would read the wrong ones. Trace a "
                "chunked input on its own."
            )
        items.extend(rows)
        forward.update(forward_kwargs)

    encoding = collate(model, items)
    supply_position_ids(model, encoding)
    return tuple(), {**encoding, **forward}


def preprocess_invoke(model, data: Any, kwargs: dict) -> tuple:
    """One invoke -> (list of per-row model-input dicts, forward kwargs).

    An input not known to be per-row — written in processor terms, a raw
    feature tensor, an encoding with a tensor outside `ROW_FIELDS`, a
    dual-encoder task's paired rows — goes to the model on its own, and comes
    back as ``(None, (args, kwargs))``: the whole model call.
    """
    media = as_processor_encoding(model, data, kwargs)
    if media is not None:
        return None, ((), media)
    if isinstance(data, torch.Tensor) and data.is_floating_point():
        return None, ((data,), kwargs)  # raw features
    # The three ways of writing model inputs are one encoding from here on,
    # so they cannot be routed differently.
    encoding = as_encoding(data, kwargs)
    if encoding is not None and (
        input_key(encoding) is None or has_nontext_keys(encoding)
    ):
        # Passed through whole. Ids written as lists (or without a batch
        # dimension) are made the tensors a forward takes first.
        if input_key(encoding) is not None and not all(
            isinstance(value, torch.Tensor) and value.dim() > 1
            for key, value in encoding.items()
            if key in ROW_FIELDS and value is not None
        ):
            rows, forward = encode_pretokenized(encoding)
            encoding = {**collate(model, rows), **forward}
        return None, ((), encoding)
    if model.task in UNTRACEABLE_TASKS:
        raise NotImplementedError(UNTRACEABLE_TASKS[model.task])
    if encoding is not None:
        return encode_pretokenized(encoding)
    # Text / image / audio: let the pipeline tokenize/featurize it, routing the
    # invoke's kwargs (truncation, chat tools, ...) through its own param split.
    preprocess_params, forward_params, _ = sanitize(model, kwargs)
    # Chat message(s) are wrapped in Chat (as Pipeline.__call__ would) so the
    # template is applied; otherwise a list of strings is one input per prompt.
    inputs = as_chats(data, chat_cls(model))
    if inputs is None:
        inputs = list(data) if isinstance(data, (list, tuple)) else [data]
        inputs = parse_task_args(model, inputs)
    rows = []
    for one in inputs:
        row = model.pipeline.preprocess(one, **preprocess_params)
        # A chunked task's preprocess is a generator: it *yields* the
        # encodings it splits one input into, each a forward of its own. They
        # are unrolled into rows, so the whole input is traced in one forward.
        rows.extend([row] if hasattr(row, "items") else row)
    if any("text_inputs" in row for row in rows):
        # A dual-encoder zero-shot task (CLIP, CLAP) nests the candidate
        # labels' text encoding in its row, and its forward pairs one
        # image/audio row with one text row per label — so it collates with
        # nothing else, and goes to the model whole.
        if len(rows) > 1:
            raise NotImplementedError(
                f"task={model.task!r} pairs each input with its own nested "
                "text encoding, so several inputs don't collate into one "
                "forward. Trace one input at a time."
            )
        row = {**rows[0], **rows[0]["text_inputs"][0]}
        tensors = {k: v for k, v in row.items() if isinstance(v, torch.Tensor)}
        return None, ((), {**tensors, **forward_params})
    return rows, forward_params


def sanitize(model, kwargs: dict) -> tuple:
    """Split an invoke's kwargs with the pipeline's ``_sanitize_parameters``,
    with generate arguments flat among the forward kwargs.

    The multimodal generating pipelines (``image-text-to-text``,
    ``any-to-any``) take generate arguments one level down, as
    ``generate_kwargs={...}``, which their own ``_forward`` unpacks; a
    ``max_new_tokens`` is folded in there too. They take any other keyword as
    a processor argument. So a ``do_sample=False`` would reach the processor,
    which ignores it, and the ``generate_kwargs`` dict would reach the model's
    ``generate``, which rejects it. Here generate arguments go to the forward
    kwargs either way, flat, as ``text-generation``'s pipeline puts them.
    """
    import inspect

    from transformers import GenerationConfig, GenerationMixin

    named = inspect.signature(model.pipeline._sanitize_parameters).parameters
    generation = {}
    if "generate_kwargs" in named:
        # max_length is also the processor's truncation length, and the
        # pipeline sends it there; leave it to the pipeline.
        names = (
            set(GenerationConfig().to_dict())
            | set(inspect.signature(GenerationMixin.generate).parameters)
        ) - set(named) - {"max_length", "self", "inputs", "kwargs"}
        generation = {k: v for k, v in kwargs.items() if k in names}
        kwargs = {k: v for k, v in kwargs.items() if k not in names}
    preprocess_params, forward_params, postprocess_params = (
        model.pipeline._sanitize_parameters(**kwargs)
    )
    forward_params = dict(forward_params)
    forward_params.update(forward_params.pop("generate_kwargs", None) or {})
    forward_params.update(generation)
    return preprocess_params, forward_params, postprocess_params


def parse_task_args(model, inputs: list) -> list:
    """Run the pipeline's ``_args_parser`` over task-input dicts.

    Some input normalization lives in the parser ``Pipeline.__call__``
    invokes, not in ``preprocess``: ``table-question-answering`` turns the
    task dict's ``table`` into the ``pd.DataFrame`` its preprocess requires
    there. Calling ``preprocess`` directly would skip it.
    """
    parser = getattr(model.pipeline, "_args_parser", None)
    if parser is None:
        return inputs
    parsed = []
    for one in inputs:
        if is_task_input(one):
            out = parser(one)
            parsed.extend(out if isinstance(out, list) else [out])
        else:
            parsed.append(one)
    return parsed


def as_processor_encoding(model, data: Any, kwargs: dict) -> Optional[dict]:
    """Run the task's processor when an invoke is written in processor terms.

    ``trace(prompt, images=[img])`` and ``trace(text=prompt, images=[img])`` name
    the *processor's* arguments, not the model's. Without this they are handed to
    the model untouched, which raises from deep inside modeling code
    (``You must specify exactly one of input_ids or inputs_embeds``) — an error
    that says nothing about the real problem. ``generate`` has always run the
    processor for these; this makes ``trace``/``scan`` agree with it.

    Returns the model-input encoding merged with any leftover forward kwargs, or
    ``None`` when this isn't a processor call and the usual routing should apply.
    """
    if model.processor is None or not (set(kwargs) & PROCESSOR_MEDIA_KEYS):
        return None

    call = {key: value for key, value in kwargs.items() if key in PROCESSOR_KEYS}
    forward = {
        key: value for key, value in kwargs.items() if key not in PROCESSOR_KEYS
    }

    if data is not None:
        if "text" in call:
            raise ValueError(
                "Got the prompt both positionally and as `text=`; pass just one."
            )
        call["text"] = data

    # Featurizing an image goes through numpy, which a fake-tensor mode refuses.
    # `scan` runs the whole batch step under one, so step outside it here: the
    # encoding is cheap, real, and `allow_non_fake_inputs` lets it into the
    # faked forward.
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        encoding = model.processor(**call, return_tensors="pt")
    return {**dict(encoding), **forward}


def collate(model, items: list) -> dict:
    """Pad per-row encodings into one batch of model-input tensors.

    A row that leaves out a field another row has gets its `default`. The
    fields of `ROW_FIELDS` are batched here: rows of one shape are
    concatenated, and narrower rows are padded with the field's own value —
    on the side the tokenizer pads on, or on the right for a field with a
    length of its own (`TARGET_FIELDS`). Anything else a row carries
    (a pipeline's image or audio features) goes to the pipeline's
    ``pad_collate_fn``, which knows the feature extractor's padding.
    """
    # Drop the pipeline's non-tensor bookkeeping (e.g. prompt_text) up front.
    items = [
        {k: v for k, v in item.items() if isinstance(v, torch.Tensor)}
        for item in items
    ]
    if len(items) == 1:
        return items[0]
    targets = TARGET_FIELDS
    if getattr(getattr(model._module, "config", None), "is_encoder_decoder", False):
        targets = targets | {"labels"}
    keys = dict.fromkeys(key for item in items for key in item)
    items = [
        {key: item[key] if key in item else default(key, item, items, targets) for key in keys}
        for item in items
    ]

    feature = model.feature_extractor or model.image_processor
    side = (
        getattr(feature, "padding_side", None)
        or getattr(model.tokenizer, "padding_side", None)
        or "right"
    )
    encoding = {}
    others = [
        {k: v for k, v in item.items() if k not in ROW_FIELDS} for item in items
    ]
    if others[0]:
        from transformers.pipelines.base import pad_collate_fn

        encoding.update(pad_collate_fn(model.tokenizer, feature)(others))
    for key in keys:
        if key in ROW_FIELDS:
            rows = [item[key] for item in items]
            encoding[key] = pad(model, key, rows, side == "left" and key not in targets)
    return encoding


def default(key: str, item: dict, items: list, targets: frozenset) -> torch.Tensor:
    """The value of ``key`` for a row that leaves it out, so it batches with ``items``.

    Segment zero, positions counted from the row's first token, and a label
    the loss ignores. A field with no default (a decoder input, a float
    label) has to be passed by every invoke or none.
    """
    present = [other[key] for other in items if key in other]
    ids = item.get("input_ids", item.get("inputs_embeds"))
    if ids is not None and key == "token_type_ids":
        return torch.zeros(ids.shape[:2], dtype=torch.long)
    if ids is not None and key == "position_ids":
        return positions(item.get("attention_mask", torch.ones(ids.shape[:2])))
    if key == "labels" and not present[0].is_floating_point():
        # Ignored labels: one per example, or one per token of this row
        # where labels follow the input (a target is widened by padding).
        like = present[0]
        if like.dim() > 1:
            aligned = ids is not None and key not in targets
            like = like[:, :1].expand(-1, ids.shape[1] if aligned else 1)
        return torch.full_like(like, -100)
    raise ValueError(
        f"Can't batch these invokes: {len(present)} of {len(items)} rows "
        f"have `{key}`, and there is no default to give the others. "
        f"Pass `{key}` in every invoke or in none."
    )


def pad(model, key: str, rows: list, left: bool) -> torch.Tensor:
    """Concatenate one field's rows, padding the narrower ones to the widest."""
    rows = [row.to(rows[0].device) for row in rows]
    if len({row.dim() for row in rows}) > 1:
        raise ValueError(
            f"Can't batch `{key}`: its rows have shapes "
            f"{', '.join(str(tuple(row.shape[1:])) for row in rows)}. "
            "Pass it the same way in every invoke."
        )
    if len({row.shape[1:] for row in rows}) > 1:
        value = ROW_FIELDS[key]
        if value is None:
            value = getattr(model.tokenizer, "pad_token_id", None)
            if value is None:
                raise ValueError(
                    f"Can't pad `{key}` rows of different lengths without a pad "
                    "token. Set `model.tokenizer.pad_token`."
                )
        width = max(row.shape[1] for row in rows)
        # `pad` counts dimensions from the last, so skip the ones after the width.
        after = (0, 0) * (rows[0].dim() - 2)
        rows = [
            torch.nn.functional.pad(
                row,
                after + ((width - row.shape[1], 0) if left else (0, width - row.shape[1])),
                value=value,
            )
            for row in rows
        ]
    return torch.cat(rows)


def positions(mask: torch.Tensor) -> torch.Tensor:
    """``position_ids`` for a ``[rows, tokens]`` attention mask.

    A row that is padding then tokens is counted from its first token, so
    left padding does not shift it. Any other row (unpadded, padded on the
    right, or masked with a gap, which is not padding) is counted from its
    first position, as the model would count it.
    """
    mask = mask.long()
    counted = (mask.cumsum(-1) - 1).clamp(min=0)
    plain = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    left_padded = (mask.diff(dim=-1) >= 0).all(-1, keepdim=True)
    return torch.where(left_padded, counted, plain)


def is_task_input(data: Any) -> bool:
    """Whether a mapping is the *task's* own input rather than model inputs.

    Some tasks take a dict — ``{"image": ..., "question": ...}`` for
    ``document-question-answering``, ``{"image": ..., "candidate_labels":
    [...]}`` for ``zero-shot-object-detection`` — which is what their
    ``preprocess`` turns into model inputs. Passed to the model as an encoding
    it fails deep in modeling code (``missing 2 required positional
    arguments``), naming nothing the caller wrote.

    Model inputs are tensors, so a mapping holding none of them is not an
    encoding: that, rather than a list of task names, is what tells the two
    apart. And it must be *tensors* specifically — a shape-duck-typed check
    misreads ``table-question-answering``'s dict, whose ``pd.DataFrame``
    table also has a ``.shape``, as an encoding.
    """
    values = list(data.values()) if isinstance(data, Mapping) else []
    return bool(values) and not any(
        isinstance(value, torch.Tensor) for value in values
    )


def has_nontext_keys(encoding: Any) -> bool:
    """Whether an encoding carries a tensor beyond the text fields of `ROW_FIELDS`.

    Tensors only: a flag riding along (``output_hidden_states=True``) is a
    forward argument, and says nothing about what the input is.
    """
    return any(
        key not in ROW_FIELDS and isinstance(value, torch.Tensor)
        for key, value in dict(encoding).items()
    )


def input_key(encoding: Mapping) -> Optional[str]:
    """Which field names the tokens, ``input_ids`` or ``inputs_embeds``; ``None`` for neither."""
    for key in ("input_ids", "inputs_embeds"):
        if encoding.get(key) is not None:
            return key
    return None


def is_token_ids(data: Any) -> bool:
    """Whether a positional input is token ids: integers, at any depth of nesting."""
    leaf = data
    while isinstance(leaf, (list, tuple)) and leaf:
        leaf = leaf[0]
    if leaf is not data and isinstance(leaf, numbers.Integral):
        return True
    if isinstance(leaf, torch.Tensor) or type(leaf).__module__ == "numpy":
        return not torch.as_tensor(leaf).is_floating_point()
    return False


def as_encoding(data: Any, kwargs: dict) -> Optional[dict]:
    """An invoke written in model inputs -> one encoding; ``None`` otherwise.

    An encoding, keywords, and ids with keywords are merged here, keywords
    winning. Text, an image, or a task's own dict is not model input.
    """
    if is_token_ids(data):
        return {"input_ids": data, **kwargs}
    mapping = kwargs if data is None else data
    # A task's own dict holds no tensors, but neither does an encoding of
    # lists: that one names its tokens.
    if not isinstance(mapping, Mapping) or (
        is_task_input(mapping) and input_key(mapping) is None
    ):
        return None
    return {**(data or {}), **kwargs}


def encode_pretokenized(encoding: dict) -> tuple:
    """A text encoding -> (one model-input dict per row, forward kwargs).

    Its `ROW_FIELDS` are split into rows, each keeping a leading batch
    dimension of 1 for `collate`; the rest are forward arguments.
    """
    fields = {
        key: value
        for key, value in encoding.items()
        if key in ROW_FIELDS and value is not None
    }
    forward = {k: v for k, v in encoding.items() if k not in ROW_FIELDS}

    source = input_key(fields)
    inputs, batched = split(source, fields.pop(source))
    items = [{source: row} for row in inputs]
    # No mask to assume beside a key-value cache: it would have to cover the
    # cached tokens too, and the model builds that one itself.
    if "attention_mask" not in fields and encoding.get("past_key_values") is None:
        for item in items:
            # One per token: ids of any rank, or embeddings less their width.
            row = item[source]
            shape = row.shape if source == "input_ids" else row.shape[:-1]
            item["attention_mask"] = torch.ones(shape, dtype=torch.long)

    width = inputs[0].shape[1]
    for key, value in fields.items():
        rows, _ = split(key, value, batched, width)
        if len(rows) == 1 and key in SHARED_FIELDS:
            rows = rows * len(items)
        if len(rows) != len(items):
            raise ValueError(
                f"`{key}` has {len(rows)} rows, but `{source}` has {len(items)}."
            )
        for item, row in zip(items, rows):
            if key in SHARED_FIELDS and row.shape[1] != item[source].shape[1]:
                raise ValueError(
                    f"`{key}` has {row.shape[1]} positions for a row of "
                    f"{item[source].shape[1]} tokens."
                )
            item[key] = row
    return items, forward


def split(
    key: str, value: Any, batched: Optional[bool] = None, width: Optional[int] = None
) -> tuple:
    """One field of an encoding -> (its rows, whether it came as a batch).

    Each row comes back with a leading batch dimension of 1. Whether the
    encoding is a batch is read off the input (``batched=None``) — ids of
    more than one dimension, or a list of sequences — and the other fields
    are split to match. Beside unbatched ids of ``width`` tokens, a flat
    value is that one row's: per-token labels, or a single class label
    (a lone value that is not one per token).
    """
    if isinstance(value, (list, tuple)) and value and not isinstance(
        value[0], numbers.Number
    ):
        # A list of sequences, which may be ragged: each entry is one row, or
        # a batch of them.
        return [row for entry in value for row in split(key, entry)[0]], True
    value = torch.as_tensor(value)
    # How many dimensions one row of the *input* has.
    row_dims = 2 if key.endswith("inputs_embeds") else 1
    if batched is None:
        batched = value.dim() > row_dims
    if value.dim() > 0 and (batched or value.dim() > row_dims):
        return list(value.unsqueeze(1)), batched
    if value.numel() == 1 and width not in (None, 1):
        return [value.reshape(1)], batched
    return [value.unsqueeze(0)], batched


def supply_position_ids(model, encoding: dict) -> None:
    """Add mask-derived ``position_ids`` for a left-padded text batch (in place).

    Left padding shifts each real token's absolute index, so an absolute-position
    model (GPT-2 family) would mispredict a short prompt padded up to a longer
    one. Counting each left-padded row from its first token (`positions`)
    keeps every real token at its true 0-based position. Applied only where
    the mask says that: a left-padding tokenizer, no positions given, a
    text-only batch, and a mask the width of the input with padding in it. A
    mask wider than the input covers a key-value cache as well, and the
    model places those tokens itself.
    """
    mask = encoding.get("attention_mask")
    # Under `scan` the forward runs on fake tensors to propagate shapes only, so
    # there are no real mask values to read -- `bool(mask.all())` would raise
    # GuardOnDataDependentSymNode. position_ids do not affect shapes, so skipping
    # the correction here changes nothing a scan can observe.
    if detect_fake_mode() is not None:
        return
    ids = encoding.get("input_ids", encoding.get("inputs_embeds"))
    if (
        "position_ids" in encoding
        or mask is None
        or ids is None
        or mask.shape != ids.shape[:2]
        or bool(mask.all())
        or getattr(model.tokenizer, "padding_side", None) != "left"
        or any(key not in ROW_FIELDS for key in encoding)
    ):
        return
    position_ids = positions(mask)
    # No left-padded row: the model's own count is already right.
    if bool((position_ids == torch.arange(mask.shape[-1], device=mask.device)).all()):
        return
    encoding["position_ids"] = position_ids

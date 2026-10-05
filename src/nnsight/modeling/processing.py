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


def batch_size(inputs: tuple, kwargs: dict) -> int:
    """Number of batch rows an invoke's input contributes.

    Accepts every input format a forward takes: a string is one row, a list of
    strings one per prompt, a single token-id list one row, and a batch of
    token-id lists / a 2-D tensor / a pre-tokenized encoding one per leading
    entry. Multimodal data passed by keyword (``text=``/``images=`` to a VLM's
    generate) counts as one row. Zero rows means params only (e.g.
    ``max_new_tokens=``), so the trace expects ``invoke()`` blocks for the data.
    """
    if inputs:
        return num_rows(inputs[0])
    if has_text_input(kwargs):
        return num_rows(kwargs)
    # A VLM generate passes its data by keyword; treat its presence as one row.
    if kwargs.get("text") is not None or any(
        kwargs.get(key) is not None
        for key in ("images", "pixel_values", "input_features")
    ):
        return 1
    return 0


def num_rows(value: Any) -> int:
    """The leading (row) dimension of one input value."""
    if isinstance(value, str):
        return 1
    if isinstance(value, Mapping) and has_text_input(value):
        key = "input_ids" if value.get("input_ids") is not None else "inputs_embeds"
        return len(split(key, value[key])[0])
    if isinstance(value, torch.Tensor):
        return 1 if value.ndim <= 1 else value.shape[0]
    chats = as_chats(value)
    if chats is not None:
        return len(chats)  # a chat conversation is one row, not one per message
    if is_token_ids(value):
        return len(split("input_ids", value)[0])
    if isinstance(value, (list, tuple)):
        # A list of strings / images is one row per element.
        return len(value)
    # A lone non-text object (e.g. a PIL image) is a single row.
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
        data = inputs[0] if inputs else None
        rows, forward_kwargs = preprocess_invoke(model, data, kwargs)
        if rows is None:
            # A raw feature tensor / multimodal encoding can't be padded into an
            # input_ids batch — pass a lone invoke straight to the model. An
            # encoding (positional or via kwargs) is unpacked as keyword inputs.
            if len(invokes) > 1:
                raise NotImplementedError(
                    "Can't batch these inputs; pass text or token ids."
                )
            # `forward_kwargs` is `kwargs` for a plain opaque input, and an
            # encoding when the invoke was written in processor terms or its
            # ids had to be made tensors.
            if forward_kwargs is not kwargs:
                return tuple(), forward_kwargs
            if isinstance(data, Mapping):
                return tuple(), {**dict(data), **kwargs}
            return inputs, kwargs
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

    Returns ``(None, kwargs)`` for an opaque input (a raw feature tensor or a
    multimodal encoding) that the caller passes through to the model untouched.
    """
    media = as_processor_encoding(model, data, kwargs)
    if media is not None:
        return None, media
    if isinstance(data, torch.Tensor) and data.is_floating_point():
        return None, kwargs  # raw features
    # The three ways of writing model inputs are one encoding from here on,
    # so they cannot be routed differently.
    encoding = as_encoding(data, kwargs)
    if encoding is not None and not has_text_input(encoding):
        return None, kwargs
    if encoding is not None and has_nontext_keys(encoding):
        # Passed through whole. Tensors go as they are; ids written as
        # lists are made the tensors a forward takes first.
        if all(
            isinstance(value, torch.Tensor) and value.dim() > 1
            for key, value in encoding.items()
            if key in ROW_FIELDS and value is not None
        ):
            return None, kwargs
        rows, forward = encode_pretokenized(encoding)
        return None, {**collate(model, rows), **forward}
    if model.task == "keypoint-matching":
        # This task's unit input is a *pair* of images, which collides with
        # the list convention (one prompt per element): the pair is split
        # into two single-image preprocess calls, and a nested pair reads
        # as pre-tokenized ids — which is why this check sits before
        # `encode_pretokenized`. (An encoding you built yourself is opaque
        # and never reaches here.)
        raise NotImplementedError(
            "task='keypoint-matching' takes a pair of images as one input, "
            "which a trace's list convention (one prompt per element) "
            "would split. Run the whole task with model.pipe([image_a, "
            "image_b]), or trace one forward on an encoding you build "
            "yourself: model.image_processor(images=[image_a, image_b], "
            "return_tensors='pt')."
        )
    if encoding is not None:
        return encode_pretokenized(encoding)
    if model.task == "mask-generation":
        # This task's preprocess *runs the model*: it embeds the image, then
        # yields one input per batch of candidate points, each carrying a copy
        # of that embedding. There is no single forward to assemble — the
        # encoder ran outside the trace, and the rows would be one copy of the
        # image embedding per point batch (128 of them at the task's default).
        raise NotImplementedError(
            "task='mask-generation' has no forward to trace from an image: its "
            "preprocess embeds the image by running the model, then yields one "
            "input per batch of candidate points. Run the whole task with "
            "model.pipe(image), or trace one forward on an encoding you build "
            "yourself: model.image_processor(image, return_tensors='pt'), with "
            "the points you want as input_points=."
        )
    # Text / image / audio: let the pipeline tokenize/featurize it, routing the
    # invoke's kwargs (truncation, chat tools, ...) through its own param split.
    preprocess_params, forward_params, _ = model.pipeline._sanitize_parameters(**kwargs)
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
        # encodings it splits one input into instead of returning one, and
        # each is a forward of its own. They are unrolled into rows here, so
        # the whole input is traced in the trace's one forward; handing the
        # generator to `collate` is what makes it ask a generator for
        # `.items()`.
        rows.extend([row] if hasattr(row, "items") else row)
    merged = [merge_nested_encodings(row) for row in rows]
    if any(row is not None for row in merged):
        # A dual-encoder zero-shot task (CLIP, CLAP) runs one forward whose
        # batch dims differ per half — one image/audio row against one text
        # row per candidate label — so its rows don't collate with anything
        # else's; a lone one goes to the model whole, like an encoding.
        if len(rows) > 1:
            raise NotImplementedError(
                f"task={model.task!r} pairs each input with its own nested "
                "text encoding, so several inputs don't collate into one "
                "forward. Trace one input at a time."
            )
        encoding = {
            key: value
            for key, value in merged[0].items()
            if isinstance(value, torch.Tensor)
        }
        return None, {**encoding, **forward_params}
    return rows, forward_params


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
        if hasattr(one, "keys") and is_task_input(one):
            out = parser(one)
            parsed.extend(out if isinstance(out, list) else [out])
        else:
            parsed.append(one)
    return parsed


def merge_nested_encodings(row: Any) -> Optional[dict]:
    """Flatten a preprocess row whose model inputs sit one level down.

    A dual-encoder zero-shot pipeline (CLIP, CLAP) returns the candidate
    labels' text encoding *nested* — ``{"pixel_values": ..., "text_inputs":
    [BatchEncoding]}`` — and unwraps it in its ``_forward`` right before
    the model call. Collation keeps only top-level tensors, which would
    silently drop the text half. Returns the row with every nested
    encoding's tensors merged in, or ``None`` when nothing is nested.
    """
    merged, found = {}, False
    for key, value in row.items():
        inner = value
        if isinstance(inner, (list, tuple)) and len(inner) == 1:
            inner = inner[0]
        if hasattr(inner, "keys") and not isinstance(inner, torch.Tensor):
            inner = dict(inner)
            if inner and all(
                isinstance(item, torch.Tensor) for item in inner.values()
            ):
                merged.update(inner)
                found = True
                continue
        merged[key] = value
    return merged if found else None


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

    The fields of `ROW_FIELDS` are batched here: rows of one shape are
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
        return dict(items[0])
    targets = target_fields(model)
    fill_missing(items, targets)

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
    for key in items[0]:
        if key in ROW_FIELDS:
            rows = [item[key] for item in items]
            encoding[key] = pad(model, key, rows, side == "left" and key not in targets)
    return encoding


def target_fields(model) -> frozenset:
    """The fields with a length of their own on this model (see `TARGET_FIELDS`)."""
    config = getattr(model._module, "config", None)
    if getattr(config, "is_encoder_decoder", False):
        return TARGET_FIELDS | {"labels"}
    return TARGET_FIELDS


def fill_missing(items: list, targets: frozenset) -> None:
    """Give every row the fields any row has (in place), so they batch.

    An invoke that leaves out a field another supplies gets a default:
    segment zero, positions counted from its first token, and a label the
    loss ignores. A field with no default (a decoder input, a float label)
    has to be passed by every invoke or none.
    """
    for key in {key for item in items for key in item}:
        present = [item[key] for item in items if key in item]
        if len(present) == len(items):
            continue
        for item in items:
            if key in item:
                continue
            ids = item.get("input_ids", item.get("inputs_embeds"))
            if ids is not None and key == "token_type_ids":
                item[key] = torch.zeros(ids.shape[:2], dtype=torch.long)
            elif ids is not None and key == "position_ids":
                item[key] = positions(
                    item.get("attention_mask", torch.ones(ids.shape[:2]))
                )
            elif key == "labels" and not present[0].is_floating_point():
                # Ignored labels: one per example, or one per token of this row
                # where labels follow the input (a target is widened by padding).
                like = present[0]
                if like.dim() > 1:
                    aligned = ids is not None and key not in targets
                    like = like[:, :1].expand(-1, ids.shape[1] if aligned else 1)
                item[key] = torch.full_like(like, -100)
            else:
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


def has_text_input(encoding: Any) -> bool:
    """Whether an encoding names the tokens: ``input_ids`` or ``inputs_embeds``."""
    return (
        encoding.get("input_ids") is not None
        or encoding.get("inputs_embeds") is not None
    )


def is_token_ids(data: Any) -> bool:
    """Whether a positional input is token ids: an integer array, or a list of ints or of sequences."""
    if isinstance(data, (list, tuple)):
        if not data or as_chats(data) is not None:
            return False
        data = data[0]
        if isinstance(data, (numbers.Integral, list, tuple)):
            return True
    if isinstance(data, torch.Tensor) or type(data).__module__ == "numpy":
        return not torch.as_tensor(data).is_floating_point()
    return False


def as_encoding(data: Any, kwargs: dict) -> Optional[dict]:
    """An invoke written in model inputs -> one encoding; ``None`` otherwise.

    An encoding, keywords, and ids with keywords are merged here, keywords
    winning. Text, an image, or a task's own dict is not model input.
    """
    if data is None:
        # Keywords alone are an encoding only if they carry an input.
        if has_text_input(kwargs) or has_nontext_keys(kwargs):
            return kwargs
        return None
    if isinstance(data, Mapping):
        # A task's own dict holds no tensors, but neither does an encoding
        # of lists: that one names its tokens.
        if is_task_input(data) and not has_text_input(data):
            return None
        return {**data, **kwargs}
    if is_token_ids(data):
        return {"input_ids": data, **kwargs}
    return None


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

    source = "input_ids" if "input_ids" in fields else "inputs_embeds"
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
    # How many dimensions one row of the *input* has.
    row_dims = 2 if key.endswith("inputs_embeds") else 1
    if isinstance(value, (list, tuple)) and value and not isinstance(
        value[0], numbers.Number
    ):
        # A list of sequences, which may be ragged: one row each, or the
        # rows of each where an entry is itself a batch.
        rows = []
        for entry in map(torch.as_tensor, value):
            rows.extend(entry.unsqueeze(1) if entry.dim() > row_dims else [entry.unsqueeze(0)])
        return rows, True
    value = torch.as_tensor(value)
    if batched is None:
        batched = value.dim() > row_dims
    if not batched and value.dim() <= row_dims:
        if value.numel() == 1 and width not in (None, 1):
            return [value.reshape(1)], batched
        return [value.unsqueeze(0)], batched
    if value.dim() == 0:
        return [value.unsqueeze(0)], batched
    return [row.unsqueeze(0) for row in value], batched


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
        or not isinstance(mask, torch.Tensor)
        or not isinstance(ids, torch.Tensor)
        or mask.dim() != 2
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

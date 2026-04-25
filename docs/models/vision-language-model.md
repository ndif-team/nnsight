---
title: VisionLanguageModel
one_liner: Extends LanguageModel with AutoProcessor support for VLMs that accept text and PIL images.
tags: [models, vlm, vision, transformers]
related: [docs/models/index.md, docs/models/language-model.md]
sources: [src/nnsight/modeling/vlm.py:17, src/nnsight/modeling/language.py:21, tests/test_vlm.py]
---

# VisionLanguageModel

## What this is for

`nnsight.VisionLanguageModel` wraps HuggingFace vision-language models (LLaVA, Qwen2-VL, InternVL, Phi-3.5-vision, etc.) — anything loadable via `AutoModelForImageTextToText` (default) plus `AutoProcessor`. It extends [`LanguageModel`](language-model.md) by:

- Loading an `AutoProcessor` that handles both text tokenization and image preprocessing
- Routing `images=` kwarg through the processor instead of the tokenizer
- Falling back to the standard text-only path when no `images=` is provided
- Batching `pixel_values` (and other image tensors) alongside `input_ids` across invokes

If your model has both text and image inputs and ships with an `AutoProcessor`, this is the wrapper.

## When to use / when not to use

Use `VisionLanguageModel` when:
- Your model is a HuggingFace VLM (LLaVA family, Qwen2-VL, Llama-3.2-Vision, Phi-3.5-vision, InternVL, MiniCPM-V, etc.).
- You want to pass PIL images alongside text and have NNsight feed them through the model's processor automatically.
- You want batching across multiple text+image invokes.

Do not use `VisionLanguageModel` when:
- Your model is text-only — use [`LanguageModel`](language-model.md). (Though `VisionLanguageModel` does fall back to text-only mode if you omit `images=`, the extra processor loading is wasted overhead.)
- Your VLM doesn't have an `AutoProcessor` — fall back to wrapping a pre-loaded HF model via `LanguageModel(hf_model, tokenizer=...)` and managing image preprocessing yourself.
- You need vLLM-grade serving — vLLM does support some VLMs but the NNsight `VLLM` integration's multi-modal coverage is limited; see `src/nnsight/modeling/vllm/IDEAS.md` for the current state.

## Loading

```python
from nnsight import VisionLanguageModel

model = VisionLanguageModel(
    "llava-hf/llava-interleave-qwen-0.5b-hf",
    device_map="auto",
    dispatch=True,
)
```

### Constructor

```python
VisionLanguageModel(
    repo_id_or_module,
    *,
    processor=None,                       # AutoProcessor instance; auto-loaded if None
    automodel=AutoModelForImageTextToText,
    tokenizer=None,                       # auto-pulled from processor.tokenizer if None
    config_model=None,
    revision=None,
    rename=None,
    envoys=None,
    dispatch=False,
    meta_buffers=True,
    import_edits=False,
    processor_kwargs=None,                # forwarded to AutoProcessor.from_pretrained
    tokenizer_kwargs=None,                # forwarded to AutoTokenizer (only if processor has none)
    **kwargs,                             # forwarded to AutoModelForImageTextToText.from_pretrained
)
```

| Parameter | Description |
|-----------|-------------|
| `processor` | A pre-loaded `AutoProcessor` instance. If `None`, loaded from the repo via `AutoProcessor.from_pretrained(repo_id, **processor_kwargs)`. See `vlm.py:101`. |
| `automodel` | The `AutoModel` class for loading. Defaults to `AutoModelForImageTextToText`. Override (e.g. with `LlavaForConditionalGeneration`) if your model isn't registered with that auto class. See `vlm.py:59`. |
| `tokenizer` | Usually omitted — `_load_processor` extracts `processor.tokenizer` automatically (`vlm.py:105`). Only relevant if you're wrapping a pre-loaded model and the processor doesn't have a tokenizer. |
| `processor_kwargs` | Forwarded to `AutoProcessor.from_pretrained`. |

All other kwargs (`device_map`, `torch_dtype`, `trust_remote_code`, etc.) work the same as on [`LanguageModel`](language-model.md).

### Dispatch behavior

Same lazy-loading semantics as `LanguageModel`. `dispatch=False` builds the model on meta tensors via `from_config`; the processor (and its tokenizer) is loaded eagerly because it's lightweight.

## Canonical pattern

```python
from nnsight import VisionLanguageModel
from PIL import Image

model = VisionLanguageModel(
    "llava-hf/llava-interleave-qwen-0.5b-hf",
    device_map="auto",
    dispatch=True,
)

img = Image.open("photo.jpg")

with model.trace("<image>\nDescribe this image", images=[img]):
    hidden = model.model.language_model.layers[-1].output.save()

print(hidden.shape)
```

The `<image>` placeholder convention is model-specific (LLaVA uses `<image>`, others differ). Use the prompt format your model expects.

### Generation

```python
with model.generate(
    "<image>\nDescribe this image",
    images=[img],
    max_new_tokens=50,
) as tracer:
    output = model.generator.output.save()

print(model.tokenizer.decode(output[0], skip_special_tokens=True))
```

### Multiple invokes with images

Each invoke gets its own image. `_batch()` (`vlm.py:237`) re-pads `input_ids` and concatenates `pixel_values` along the batch dimension.

```python
img1 = Image.open("a.jpg")
img2 = Image.open("b.jpg")

with model.trace() as tracer:
    with tracer.invoke("<image>\nDescribe image one", images=[img1]):
        out_a = model.lm_head.output[:, -1].save()

    with tracer.invoke("<image>\nDescribe image two", images=[img2]):
        out_b = model.lm_head.output[:, -1].save()

    with tracer.invoke():                          # empty invoke — full batch
        out_all = model.lm_head.output[:, -1].save()
```

### Text-only fallback

When no `images=` kwarg is provided, `_prepare_input` (`vlm.py:160`) delegates entirely to the parent `LanguageModel` path:

```python
with model.trace("Hello world"):                   # no images, no processor call
    hidden = model.model.language_model.layers[-1].output.save()
```

### Modifying activations

```python
with model.trace("<image>\nDescribe", images=[img]):
    pre = model.model.language_model.layers[-1].output.clone().save()
    model.model.language_model.layers[-1].output[:] = 0
    post = model.model.language_model.layers[-1].output.save()
```

Note that LLaVA's underlying Qwen2 decoder layers return a tensor directly (not a tuple), so `.output` is `[batch, seq, hidden]`. GPT-2-style blocks return `(hidden, ...)` tuples; check your model's forward signature.

## Special properties

| Attribute | Description | Source |
|-----------|-------------|--------|
| `model.processor` | The `AutoProcessor`. Handles both text and images. Use `model.processor(text=..., images=...)` directly if you want to preprocess outside a trace. | `vlm.py:63`, `vlm.py:101` |
| `model.tokenizer` | The processor's tokenizer (`processor.tokenizer`). Not loaded separately. | `vlm.py:105` |
| `model.generator` | Same as `LanguageModel` — captures `.generate()` output. |
| `model.config`, `model.automodel`, `model.repo_id`, `model.revision`, `model.dispatched`, `model._model` | All inherited from `LanguageModel` / `TransformersModel`. |

## Limitations

- **Processor must accept `text=` and `images=` kwargs.** This is the standard HF `AutoProcessor` interface. If your model has a custom processor with a different signature, override `_prepare_input` or pass already-preprocessed inputs as a dict.
- **One image set per invoke.** Pass either a single PIL image or a list of PIL images. Token-ID inputs go through the parent `LanguageModel` path and `images` is handled separately via the `**kwargs` dict.
- **Batching of mixed image/no-image invokes.** `_batch` concatenates image tensors along dim 0. Mixing invokes with and without images in the same trace can produce shape mismatches if the model's processor expects all-or-nothing.
- **Chat-template prompts.** If your VLM uses a structured chat template (Qwen2-VL, Llama-3.2-Vision, etc.), build the prompt with `processor.apply_chat_template(...)` before passing it in, or pass an already-tokenized dict.

## Gotchas

- **Default `automodel` is `AutoModelForImageTextToText`.** Some older VLMs (early LLaVA) aren't registered with this class. Either upgrade to a `-hf` repo variant or pass `automodel=LlavaForConditionalGeneration` explicitly.
- **`<image>` placeholder is model-specific.** LLaVA expects `<image>` in the prompt string. Qwen2-VL uses `<|vision_start|><|image_pad|><|vision_end|>`. Use whatever format the model card prescribes.
- **Tuple-vs-tensor outputs vary by backbone.** Many VLM language backbones (Qwen2 inside LLaVA-Interleave) return tensors directly from decoder layers; GPT-2-style backbones return tuples. Check shapes in `.scan()` first.
- **Empty invoke after image invoke** sees the full padded batch (including `pixel_values`). See `tests/test_vlm.py:185` for the expected behavior.

## Related

- [docs/models/language-model.md](language-model.md) — parent class, all `LanguageModel` features apply
- [docs/models/index.md](index.md) — full decision tree
- `src/nnsight/modeling/vlm.py` — source
- `tests/test_vlm.py` — runnable examples covering trace, generate, batching, text-only fallback, scan

---
title: VisionLanguageModel (deprecated)
one_liner: Deprecated thin alias for TransformersModel(task="image-text-to-text"); warns on construction.
tags: [models, vlm, vision, transformers, deprecated]
related: [docs/models/transformers-model.md, docs/models/language-model.md, docs/models/index.md]
sources: [src/nnsight/modeling/vlm.py:29, src/nnsight/modeling/transformers.py:161, tests/test_vlm.py]
---

# VisionLanguageModel (deprecated)

> **Deprecated.** `VisionLanguageModel` is now a thin subclass of [`TransformersModel`](transformers-model.md) that pins `task="image-text-to-text"` and warns on construction. Use `TransformersModel(repo_id, task="image-text-to-text")` instead.

Constructing one emits a `DeprecationWarning`:

```
VisionLanguageModel is deprecated; use TransformersModel(repo_id, task='image-text-to-text') instead.
```

## Migration

```python
# OLD
from nnsight import VisionLanguageModel
model = VisionLanguageModel("llava-hf/llava-1.5-7b-hf", dispatch=True)

# NEW
from nnsight import TransformersModel
model = TransformersModel("llava-hf/llava-1.5-7b-hf", task="image-text-to-text", dispatch=True)
```

## The one behavioral difference: `generate`

The deprecated alias keeps its old `generate` shape — it runs the **processor** itself over the prompt and images, then generates through the model, returning **token ids** (`vlm.py:52`). Images and text go by keyword (`text=`, `images=`), as older nnsight took them:

```python
from nnsight import VisionLanguageModel

model = VisionLanguageModel("trl-internal-testing/tiny-LlavaForConditionalGeneration", dispatch=True)

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe"}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)

with model.generate(text=prompt, images=[img], max_new_tokens=3, do_sample=False) as tracer:
    ids = tracer.result.save()
print(model.tokenizer.batch_decode(ids))
```

The prompt may be positional or `text=`; both are equivalent (`tests/test_vlm.py:207`). Batching multiple multimodal generate inputs is **not** supported — use one invoke.

### Doing the same with `TransformersModel`

On a plain `TransformersModel(task="image-text-to-text")`, `trace`/`scan`/`generate` all take the prompt plus `images=` directly — the processor runs for you:

```python
model = TransformersModel(REPO, task="image-text-to-text", dispatch=True)

with model.trace(prompt, images=[img]):          # or trace(text=prompt, images=[img])
    projected = model.model.multi_modal_projector.output.save()
    logits = model.output.logits.save()
```

Any invoke naming a media argument (`images`, `audio`, `videos`, ...) is treated as a **processor call**: the processor runs over it and the resulting encoding goes to the model. Passing the prompt both positionally and as `text=` is an error. Anything else in the invoke (`output_hidden_states=True`, ...) is passed through to the forward.

Running the processor yourself still works and is identical — useful when you want the encoding for other reasons (say, to read `input_ids` for position labels):

```python
enc = model.processor(images=img, text=prompt, return_tensors="pt")
with model.trace(enc):                            # or trace(**enc)
    logits = model.output.logits.save()
```

A multimodal encoding (carrying `pixel_values`) is routed to the model untouched rather than re-tokenized as text (`transformers.py:759`, `tests/test_vlm.py:148`).

`pipe` remains the way to get the pipeline's decoded records rather than raw tensors:

```python
with model.pipe(text=prompt, images=img, max_new_tokens=3, do_sample=False) as tracer:
    records = tracer.result.save()       # [{'generated_text': ...}]
```

## What still works

Everything inherited from `TransformersModel` — `trace`, `scan`, interventions on the vision tower / projector / language model, `model.generator.output`, iteration, gradients (see `tests/test_vlm.py`). `model.processor` and `model.tokenizer` (derived from the processor) are populated for image-text-to-text.

## Related

- [docs/models/transformers-model.md](transformers-model.md) — the class to use instead
- [docs/models/language-model.md](language-model.md) — the text-only equivalent (also deprecated)
- [docs/models/index.md](index.md) — decision tree
- `src/nnsight/modeling/vlm.py` — source
- `tests/test_vlm.py` — runnable examples (trace, generate, pipe, scan, input routing)

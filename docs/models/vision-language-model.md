---
title: VisionLanguageModel (deprecated)
one_liner: Deprecated alias for TransformersModel(task="image-text-to-text"); warns on construction.
tags: [models, vlm, vision, transformers, deprecated]
related: [docs/models/transformers-model.md, docs/models/language-model.md, docs/models/index.md]
sources: [src/nnsight/modeling/vlm.py, tests/test_vlm.py]
---

# VisionLanguageModel (deprecated)

> **Deprecated.** `VisionLanguageModel` is a [`TransformersModel`](transformers-model.md) with the task pinned to `"image-text-to-text"`. Use `TransformersModel(repo_id, task="image-text-to-text")`.

Constructing one warns:

```
NNsightDeprecationWarning: VisionLanguageModel is deprecated; use
TransformersModel(repo_id, task='image-text-to-text') instead.
```

`NNsightDeprecationWarning` is a `FutureWarning`, so the warning reaches you wherever the construction lives, not only in `__main__`.

## Migration

```python
# OLD
from nnsight import VisionLanguageModel
model = VisionLanguageModel("llava-hf/llava-1.5-7b-hf", dispatch=True)

# NEW
from nnsight import TransformersModel
model = TransformersModel("llava-hf/llava-1.5-7b-hf", task="image-text-to-text", dispatch=True)
```

The task string is exactly `"image-text-to-text"`. Everything else carries over unchanged — `trace`, `scan`, `pipe`, `model.processor`, `model.tokenizer`, and interventions on the vision tower, the projector and the language stack.

One behavior does not carry over. `VisionLanguageModel.generate` runs the processor itself over `text=` and `images=` and returns generated **token ids**:

```python
with model.generate(text=prompt, images=[img], max_new_tokens=3, do_sample=False) as tracer:
    ids = tracer.result.save()
print(model.tokenizer.batch_decode(ids))
```

On `TransformersModel(task="image-text-to-text")`, `trace` / `scan` / `generate` take the same prompt and `images=` and the processor still runs for you, so the call above needs no change beyond the constructor. Running the processor yourself and passing the encoding is equivalent, and gives you the `input_ids` to compute image-token positions from:

```python
encoding = model.processor(images=img, text=prompt, return_tensors="pt")
with model.trace(encoding):
    logits = model.output.logits.save()
```

Neither form batches across invokes: an encoding carrying `pixel_values` cannot be padded into a batch, and a second invoke raises `NotImplementedError: Can't batch these inputs; pass text or token ids.`

## Related

- [docs/models/transformers-model.md](transformers-model.md) — the class to use instead
- [docs/models/language-model.md](language-model.md) — the text-only equivalent, also deprecated
- [docs/models/index.md](index.md) — decision tree
- `src/nnsight/modeling/vlm.py` — source
- `tests/test_vlm.py` — runnable examples (trace, generate, pipe, scan, input routing)

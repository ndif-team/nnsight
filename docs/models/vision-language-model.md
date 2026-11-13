---
title: VisionLanguageModel (deprecated)
one_liner: Deprecated alias for TransformersModel(task="image-text-to-text"); warns on construction.
tags: [models, vlm, vision, transformers, deprecated]
related: [docs/models/transformers-model.md, docs/models/language-model.md, docs/models/index.md]
sources: [src/nnsight/modeling/vlm.py, tests/test_vlm.py]
---

# VisionLanguageModel (deprecated)

> **Deprecated.** `VisionLanguageModel` is a [`TransformersModel`](transformers-model.md) with the task pinned to `"image-text-to-text"`. Use `TransformersModel(repo_id, task="image-text-to-text")`.

Constructing one emits `nnsight.NNsightDeprecationWarning` (a `FutureWarning`, so it shows wherever the construction happens):

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

Only the constructor changes. The alias overrides nothing but the task, so `trace`, `scan`, `pipe`, `model.processor`, `model.tokenizer`, and `generate` over `text=` and `images=` returning token ids all behave the same on both:

```python
with model.generate(text=prompt, images=[img], max_new_tokens=3, do_sample=False) as tracer:
    ids = tracer.result.save()
print(model.tokenizer.batch_decode(ids))
```

Input forms (chat messages with inline images, a processor encoding), generation keywords, the image-token span and which VLM inputs batch across invokes are covered in [transformers-model.md](transformers-model.md#vision-language-models).

## Related

- [docs/models/transformers-model.md](transformers-model.md) — the class to use instead
- [docs/models/language-model.md](language-model.md) — the text-only equivalent, also deprecated
- [docs/models/index.md](index.md) — decision tree
- `src/nnsight/modeling/vlm.py` — source
- `tests/test_vlm.py` — runnable examples (trace, generate, pipe, scan, input routing)

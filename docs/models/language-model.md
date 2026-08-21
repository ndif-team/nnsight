---
title: LanguageModel (deprecated)
one_liner: Deprecated thin alias for TransformersModel(task="text-generation"); warns on construction.
tags: [models, language, transformers, deprecated]
related: [docs/models/transformers-model.md, docs/models/index.md, docs/models/vision-language-model.md]
sources: [src/nnsight/modeling/language.py:19, src/nnsight/modeling/transformers.py:161, tests/test_language.py:33]
---

# LanguageModel (deprecated)

> **Deprecated.** `LanguageModel` is now a thin subclass of [`TransformersModel`](transformers-model.md) that pins `task="text-generation"` and warns on construction. Use `TransformersModel(repo_id, task="text-generation")` instead. Everything this class offered — `generate` returning token ids, `pipe` running the pipeline, `trace`, `scan` — lives on `TransformersModel` now.

Constructing one emits a `DeprecationWarning` (verified):

```
LanguageModel is deprecated; use TransformersModel(repo_id, task='text-generation') instead.
```

## Migration

```python
# OLD
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", dispatch=True)

# NEW
from nnsight import TransformersModel
model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
```

The one thing the alias still adds is `tokenizer_kwargs=` at load (applied to `model.tokenizer` after loading). With `TransformersModel`, set those on the tokenizer yourself:

```python
# LanguageModel convenience
model = LanguageModel("gpt2", tokenizer_kwargs={"padding_side": "right"})

# equivalent with TransformersModel
model = TransformersModel("gpt2", task="text-generation")
model.tokenizer.padding_side = "right"
```

## What still works

The deprecated class behaves exactly as `TransformersModel(task="text-generation")` (see `tests/test_language.py:33`):

```python
from nnsight import LanguageModel

model = LanguageModel("gpt2", dispatch=True)      # DeprecationWarning
PROMPT = "Madison Square Garden is located in the city of"

# generate -> token ids (greedy by default), read off tracer.result
with model.generate(PROMPT, max_new_tokens=3) as tracer:
    ids = tracer.result.save()
assert ids.shape == (1, 12)
print(model.tokenizer.decode(ids[0]))   # "...the city of New York City"

# trace -> one forward
with model.trace(PROMPT):
    hidden = model.transformer.h[-1].output.save()
```

- `generate` returns **token ids** (old nnsight's `generate` returned decoded records — that is now `model.pipe(...)`).
- `model.generator.output` and `model.generator.streamer.output` behave as on `TransformersModel`.
- Batching, iteration (`tracer.iter`), gradients, caching, sessions, and renaming all inherit unchanged.

## Remote

`LanguageModel` shares `TransformersModel`'s remote key (`language.py:51`), so a model deployed as a `TransformersModel` is reachable when wrapped as a `LanguageModel` (and vice versa).

## Related

- [docs/models/transformers-model.md](transformers-model.md) — the class to use instead
- [docs/models/vision-language-model.md](vision-language-model.md) — the VLM equivalent (also deprecated)
- [docs/models/index.md](index.md) — decision tree
- `src/nnsight/modeling/language.py` — source

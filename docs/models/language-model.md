---
title: LanguageModel (deprecated)
one_liner: Deprecated alias for TransformersModel(task="text-generation"); warns on construction.
tags: [models, language, transformers, deprecated]
related: [docs/models/transformers-model.md, docs/models/index.md, docs/models/vision-language-model.md]
sources: [src/nnsight/modeling/language.py, tests/test_language.py]
---

# LanguageModel (deprecated)

`nnsight.LanguageModel` was the wrapper for causal and masked language models. It
is now a thin subclass of [`TransformersModel`](transformers-model.md) that pins
`task="text-generation"` and warns on construction. Use `TransformersModel`
directly:

```python
# OLD
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", dispatch=True)

# NEW
from nnsight import TransformersModel
model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
```

`task="text-generation"` covers causal LMs and chat templating; a masked model
wants `task="fill-mask"`. Construction emits `nnsight.NNsightDeprecationWarning`
(a `FutureWarning`, so it shows wherever the construction happens):

```
LanguageModel is deprecated; use TransformersModel(repo_id, task='text-generation') instead.
```

The alias's `tokenizer_kwargs=` becomes one line on `TransformersModel`:
`model.tokenizer.padding_side = "right"`.

Remotely, the alias uses `TransformersModel`'s model key with the task pinned to
`text-generation`. The task is part of the key, so it reaches a deployment only
when that deployment's task is `text-generation` (see
[transformers-model.md](transformers-model.md#remote)).

## Related

- [docs/models/transformers-model.md](transformers-model.md) — the class to use instead
- [docs/models/vision-language-model.md](vision-language-model.md) — the VLM equivalent, also deprecated
- [docs/models/index.md](index.md) — decision tree

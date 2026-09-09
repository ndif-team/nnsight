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

The task string is `"text-generation"`, which covers causal LMs and the
pipeline's chat templating. A masked model wants `task="fill-mask"`.

Constructing one emits `nnsight.NNsightDeprecationWarning`, a `FutureWarning`, so
it is shown wherever the construction happens rather than only in `__main__`:

```
LanguageModel is deprecated; use TransformersModel(repo_id, task='text-generation') instead.
```

Everything the class offered is on `TransformersModel`: `trace`, `scan`,
`generate` returning token ids, `pipe` running the whole pipeline, batching,
`tracer.iter`, gradients, caching, sessions and renaming. The one convenience it
adds is `tokenizer_kwargs=` at load, which becomes one line afterwards:

```python
model = TransformersModel("openai-community/gpt2", task="text-generation")
model.tokenizer.padding_side = "right"
```

Both classes share a remote model key, so a checkpoint deployed as a
`TransformersModel` is reachable when wrapped as either.

## Related

- [docs/models/transformers-model.md](transformers-model.md) — the class to use instead
- [docs/models/vision-language-model.md](vision-language-model.md) — the VLM equivalent, also deprecated
- [docs/models/index.md](index.md) — decision tree

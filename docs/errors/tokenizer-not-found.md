---
title: Tokenizer Not Found
one_liner: "AttributeError: Tokenizer not found — pre-loaded model wrapped by LanguageModel without a tokenizer."
tags: [error, setup, language-model]
related: [docs/models/language-model.md, docs/usage/trace.md]
sources: [src/nnsight/modeling/language.py:204]
---

# Tokenizer Not Found

## Symptom

```
AttributeError: Tokenizer not found. If you passed a pre-loaded model to `LanguageModel`, you need to provide a tokenizer when initializing: `LanguageModel(model, tokenizer=tokenizer)`.
```

## Cause

`LanguageModel._tokenize` (`src/nnsight/modeling/language.py:200`) needs a tokenizer to convert string input into token ids:

```python
if self.tokenizer is None:
    if self.repo_id is not None:
        self._load_tokenizer(self.repo_id, **kwargs)
    else:
        raise AttributeError(
            "Tokenizer not found. If you passed a pre-loaded model to "
            "`LanguageModel`, you need to provide a tokenizer when initializing: "
            "`LanguageModel(model, tokenizer=tokenizer)`."
        )
```

`LanguageModel` only auto-loads a tokenizer when it was constructed from a HuggingFace **repo id string**. When you pass an already-loaded `nn.Module` instance there is no repo id to look up, so the tokenizer must be supplied explicitly.

## Common triggers

- Building the model yourself and wrapping it: `LanguageModel(AutoModelForCausalLM.from_pretrained(...))` without `tokenizer=`.
- Passing a custom `nn.Module` that isn't a HuggingFace model to `LanguageModel`.
- Loading the wrapper from a checkpoint that didn't persist `repo_id` and re-using it with string inputs.

## Fix

```python
# WRONG — pre-loaded model, no tokenizer supplied
from transformers import AutoModelForCausalLM
from nnsight import LanguageModel
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = LanguageModel(hf_model)
with model.trace("Hello"):                   # AttributeError on tokenize
    ...
```

```python
# FIXED — pass tokenizer alongside the pre-loaded model
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")
model = LanguageModel(hf_model, tokenizer=tok)
with model.trace("Hello"):
    ...
```

```python
# ALSO FIXED — let LanguageModel load both from the repo id
model = LanguageModel("gpt2")                # tokenizer auto-loaded
```

If you must keep the pre-loaded model and have no tokenizer at all, you can still pass tensor input directly (the `_tokenize` path short-circuits on `torch.Tensor`):

```python
input_ids = torch.tensor([[1, 2, 3]])
with model.trace(input_ids):
    ...
```

## Mitigation / how to avoid

- When wrapping a pre-loaded model, always pair it with a tokenizer in the constructor.
- If you need to subclass `LanguageModel` for a custom architecture, override `_load_tokenizer` so future `repo_id`-based loads work.
- For tensor-only pipelines (already-tokenized data), pass tensors and the error never fires.

## Related

- `src/nnsight/modeling/language.py:200`
- `docs/models/language-model.md`
- `docs/usage/trace.md`

---
title: Batching Is Not Implemented
one_liner: "NotImplementedError: Batching is not implemented for this model — multiple input invokes used on a model without _prepare_input/_batch."
tags: [error, batching, setup]
related: [docs/usage/invoke-and-batching.md, docs/concepts/batching-and-invokers.md]
sources: [src/nnsight/intervention/batching.py:104, src/nnsight/intervention/batching.py:181]
---

# Batching Is Not Implemented

## Symptom

```
NotImplementedError: Batching is not implemented for this model. Multiple invokers with inputs require `_prepare_input()` and `_batch()` methods on your model class (see `LanguageModel` for a reference implementation). Without these methods, you can still use one invoke with input and additional empty invokes (no arguments) — empty invokes operate on the entire batch and are useful for breaking up interventions to avoid execution-order conflicts.
```

## Cause

`Batchable._batch` raises `NotImplementedError` by default (`src/nnsight/intervention/batching.py:104`). The `Batcher` calls it from `Batcher.batch` whenever a **second (or later) input invoke** is registered (`:181`):

```python
self.batched_args, self.batched_kwargs = batchable._batch(
    (self.batched_args, self.batched_kwargs), *args, **kwargs
)
```

The first input invoke just stores its prepared input directly — it never touches `_batch`. So the error is raised only when you try to merge **two or more inputs into a single forward pass**.

`LanguageModel` overrides `_prepare_input` (tokenizes) and `_batch` (pads + concatenates) so it just works. Bare `NNsight(my_torch_module)` inherits the default implementation and cannot batch arbitrary tensor inputs without help.

## Common triggers

- Multiple `tracer.invoke(...)` calls with different inputs on a `model = NNsight(some_torch_module)`.
- Wrapping a HuggingFace model directly with `NNsight(hf_model)` instead of `LanguageModel(hf_model, tokenizer=tok)`.
- Custom subclasses that override `_prepare_input` but forget to override `_batch` (or vice versa).

## Fix

Three options, in order of preference.

**Option 1 — Use `LanguageModel` if it's a HuggingFace LM:**

```python
# WRONG — bare NNsight doesn't know how to batch tokens
from nnsight import NNsight
from transformers import AutoModelForCausalLM
model = NNsight(AutoModelForCausalLM.from_pretrained("gpt2"))
with model.trace() as tracer:
    with tracer.invoke("Hello"): ...
    with tracer.invoke("World"): ...    # NotImplementedError
```

```python
# FIXED — LanguageModel implements _prepare_input + _batch
from nnsight import LanguageModel
model = LanguageModel("gpt2")
with model.trace() as tracer:
    with tracer.invoke("Hello"): ...
    with tracer.invoke("World"): ...
```

**Option 2 — One input invoke + empty invokes:**

Empty invokes (no arguments) skip `_batch` entirely (`src/nnsight/intervention/batching.py:196`) and operate on whatever the first input invoke already batched. This works even on bare `NNsight`.

```python
with model.trace() as tracer:
    with tracer.invoke(input_tensor):         # one input invoke
        a = model.layer_5.output.save()
    with tracer.invoke():                     # empty invoke — full batch, new thread
        b = model.layer_2.output.save()
```

**Option 3 — Implement `_prepare_input` and `_batch` on your model class:**

```python
class MyModel(NNsight):
    def _prepare_input(self, *args, **kwargs):
        # turn raw user input into (args, kwargs, batch_size)
        x = args[0]
        return (x,), kwargs, x.shape[0]

    def _batch(self, batched_input, *args, **kwargs):
        (b_args, b_kwargs) = batched_input
        merged = torch.cat([b_args[0], args[0]], dim=0)
        return (merged,) + b_args[1:], b_kwargs
```

See `LanguageModel` (`src/nnsight/modeling/language.py`) for a real-world reference implementation.

## Mitigation / how to avoid

- Reach for `LanguageModel` whenever the underlying model is a HuggingFace causal LM — batching is already wired.
- If you only need to break interventions across multiple "passes" (not actually batch new inputs), use one input invoke + empty invokes.
- When subclassing `NNsight` for a non-LM model, implement both `_prepare_input` and `_batch` together.

## Related

- `docs/usage/invoke-and-batching.md`
- `docs/concepts/batching-and-invokers.md`
- `src/nnsight/modeling/language.py` (reference implementation)

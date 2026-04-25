---
title: Edit
one_liner: Persistently install interventions on a model with `model.edit()`; clear with `model.clear_edits()`.
tags: [usage, edit, persistent]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/usage/skip.md]
sources: [src/nnsight/intervention/tracing/editing.py, src/nnsight/intervention/backends/editing.py, src/nnsight/intervention/envoy.py:316, src/nnsight/modeling/huggingface.py:65]
---

# Edit

## What this is for

`model.edit()` opens an editing context whose body — the same intervention DSL as a regular trace — is **compiled and stored as a default mediator on the Envoy** instead of being executed once. Every subsequent `model.trace(...)` / `model.generate(...)` runs the stored interventions in addition to whatever the user writes.

Two flavors:

- `model.edit()` (default) — non-inplace. Returns a shallow copy of the root envoy with the new mediator attached. The original `model` is unchanged.
- `model.edit(inplace=True)` — mutates the original model. All future traces on `model` see the edit.

## When to use / when not to use

- Use to install always-on transforms (zero a head, add a steering vector, swap an SAE in for an MLP, etc.) without rewriting every trace.
- Use `inplace=True` when you want every consumer of `model` to see the edit (and you can serialize / `export_edits` it for reuse).
- Use the non-inplace form when you want to A/B compare original vs edited.
- Don't use for one-off interventions — that is what `model.trace(...)` is for.

## Canonical pattern

```python
# Non-inplace: returns a shallow-copied edited model
with model.edit() as edited_model:
    edited_model.transformer.h[1].output[0][:, 1] = 0

with model.trace("Hello"):
    out_original = model.transformer.h[1].output[0].save()

with edited_model.trace("Hello"):
    out_edited = edited_model.transformer.h[1].output[0].save()
```

## In-place editing

```python
with model.edit(inplace=True):
    model.transformer.h[1].output[0][:] = 0

# Now every trace through model uses the edit
with model.trace("Hello"):
    out = model.transformer.h[1].output[0].save()  # zeros
```

## Clearing edits

```python
model.clear_edits()      # drops all stored mediators on this Envoy
```

`clear_edits` simply does `self._default_mediators = []` (`envoy.py:350`).

## How it works

`Envoy.edit(...)` returns an `EditingTracer` (`src/nnsight/intervention/tracing/editing.py:15`). The tracer captures the with-block body just like an `InterleavingTracer`, but its backend is `EditingBackend` (`src/nnsight/intervention/backends/editing.py:11`):

```python
class EditingBackend(Backend):
    def __call__(self, tracer):
        invoker = tracer.invoke()
        invoker.info = tracer.info.copy()
        fn = super().__call__(invoker)
        mediator = Mediator(fn, invoker.info)
        tracer.model._default_mediators = (
            tracer.model._default_mediators + [mediator]
        )
```

In `InterleavingTracer.compile` (`tracing/tracer.py:344`), every new tracer prepends `_default_mediators` to its own mediators list, so the stored interventions run **before** the user's invokes.

## Multiple edits stack

```python
with model.edit(inplace=True):
    model.transformer.h[0].output[0][:] = 0   # first edit

with model.edit(inplace=True):
    model.transformer.h[1].output[0][:] = 0   # second edit, both apply
```

Each `edit` appends a new `Mediator` to `_default_mediators`. They run in registration order on every subsequent trace.

## Persisting edits to disk (HuggingFaceModel only)

`HuggingFaceModel` (and its subclasses including `LanguageModel`) adds `export_edits` / `import_edits` (`src/nnsight/modeling/huggingface.py:65`). They wrap the base `Envoy.export_edits` / `Envoy.import_edits` (`src/nnsight/intervention/envoy.py:356`):

```python
# After making in-place edits
model.export_edits(variant="zeroed_layers")

# Later, in a fresh process
model = LanguageModel("openai-community/gpt2")
model.import_edits(variant="zeroed_layers")
```

`export_edits` serializes `self._default_mediators` via the source-based serializer (`src/nnsight/intervention/serialization.py`) into the HF cache under `nnsight/exports/<repo>/<variant>.dill`. `import_edits` deserializes them back. Auto-import on load is supported via the `import_edits` constructor kwarg:

```python
LanguageModel("openai-community/gpt2", import_edits=True)             # __default__
LanguageModel("openai-community/gpt2", import_edits="zeroed_layers")  # named variant
```

## Gotchas

- Default mediators run **first** in every subsequent trace, before user invokes. Their interventions are visible to user code.
- Non-inplace `edit()` returns a `_shallow_copy` of the envoy — the underlying `torch.nn.Module` is shared. Edits do not duplicate weights.
- Calling `export_edits` before any `edit()` raises `ValueError: Cannot export an Envoy before calling .edit().`
- Importing edits replaces nothing — they append to existing `_default_mediators`. Call `clear_edits` first if you want a clean slate.
- The non-inplace return is the **edited model envoy** (not a tracer) — `with model.edit() as edited_model:` binds the model.

## Related

- `docs/usage/trace.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/skip.md`
- `docs/usage/save.md`

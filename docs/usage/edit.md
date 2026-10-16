---
title: Edit
one_liner: Persistently install interventions on a model with `model.edit(inplace=...)`; clear with `model.clear_edits()`.
tags: [usage, edit, persistent]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/usage/skip.md, docs/usage/iteration.md]
sources: [src/nnsight/intervention/editing.py, src/nnsight/intervention/envoy.py]
---

# Edit

`model.edit(...)` opens a context whose body — the same intervention code as a
trace — is **stored on the envoy** instead of run once. Every later
`model.trace(...)` / `generate(...)` / `pipe(...)` replays the stored
interventions **first**, before your invokes, so their effects are visible to your
code. Use it for always-on transforms (zero a head, add a steering vector, attach
an SAE or LoRA) without rewriting every trace; for a one-off intervention use a
trace.

- `model.edit()` (default, `inplace=False`) stores the edit on a **shallow copy**
  and binds `(tracer, edited)`; `model` stays clean, which suits A/B comparisons.
- `model.edit(inplace=True)` stores it on the envoy itself and binds only
  `tracer`; every consumer of `model` sees it.

## Canonical pattern (non-inplace)

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

# Store the edit on a copy; `model` stays clean.
with model.edit() as (tracer, edited):
    edited.transformer.h[0].output[:] = 0

with edited.trace("Hello world"):
    out_edited = edited.transformer.h[0].output.save()    # zeros
with model.trace("Hello world"):
    out_original = model.transformer.h[0].output.save()   # unchanged

# len(model._edits) == 0 ; len(edited._edits) == 1
```

## In-place editing

```python
with model.edit(inplace=True) as tracer:
    model.transformer.h[1].output[:] = 0

# Now every trace through model replays the edit
with model.trace("Hello world"):
    out = model.transformer.h[1].output.save()            # zeros
```

## Clearing edits

```python
model.clear_edits()          # drops all stored edits on this envoy (sets _edits = [])
```

The non-inplace copy shares the underlying `torch.nn.Module`, interleaver and
children; only the stored edits are independent, so no weights are duplicated.

## Multiple edits stack

```python
with model.edit(inplace=True):
    model.transformer.h[0].output[:] = 0     # first edit
with model.edit(inplace=True):
    model.transformer.h[1].output[:] = 0     # second edit — both apply

# len(model._edits) == 2 ; they run in registration order on every trace
```

## Attaching a module in an edit

An edit can attach a module to the tree (adapter/LoRA/SAE) and route activations through it with `hook=True`, so the trace watches it ([access-and-modify](access-and-modify.md#calling-modules-directly-inside-a-trace)):

```python
model.transformer.h[0].adapter = MyAdapter()
with model.edit(inplace=True):
    acts = model.transformer.h[0].output
    model.transformer.h[0].output = model.transformer.h[0].adapter(acts, hook=True)
```

Route it as a **replacement** (`output = ...`), not an in-place write (`output[:] = ...`).
The in-place form writes the attachment's result back into the tensor that is its own
input, so autograd refuses the later backward with `RuntimeError: one of the variables
needed for gradient computation has been modified by an inplace operation` — which is
exactly the training the attachment exists for.

A plain edit applies once (at the location's first occurrence). To re-apply it at every occurrence — each step of a generation loop — put the passthrough under the tracer's `iter`:

```python
with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:
        acts = model.transformer.h[0].output
        model.transformer.h[0].output = model.transformer.h[0].adapter(acts, hook=True)
```

## Which envoy an edit belongs to

An edit is stored on **the envoy you called `.edit()` on**, and a trace replays
only the edits of the envoy you called `.trace()` on. So an edit made with
`model.transformer.h[9].edit(...)` does not run in `model.trace(...)` (it runs in
`model.transformer.h[9].trace(hidden_states)`), and `model.clear_edits()` does
not remove it; `model.transformer.h[9].clear_edits()` does. To scope an edit to
one layer of a whole-model run, store it on the model and write the layer's path
inside:

```python
with model.edit(inplace=True):
    model.transformer.h[9].output[:] = 0
```

## Edits apply through the interleaver, not to the module

Everything that goes through a tracing call replays the stored edits, including
`trace=False`. A direct call on the wrapped `torch.nn.Module` does not:

| call | edit applies |
|---|---|
| `model.trace(...)` | yes |
| `model.generate(...)` | yes, at prefill only unless under `tracer.iter` |
| `model.pipe(...)` | yes |
| `model.trace(**ids, trace=False)` | yes |
| `model(**ids)` | **no** |
| `model._module(**ids)` | **no** |

```python
with model.edit(inplace=True):
    model.transformer.h[9].output[:] = 0

model.trace(**ids, trace=False).logits[0, -1].argmax()   # ','      edited
model(**ids).logits[0, -1].argmax()                      # ' Paris' unedited
```

So anything holding the underlying module — a HuggingFace `Trainer`, an eval
harness, `generate()` called on `model._module` — sees the model as it was. Route
those through `model.trace(...)`, or make the change in the weights instead.

## Remote edits

Edits ride with the model to a remote server — they live in `envoy._edits`, which serializes by value:

```python
with model.edit() as (tracer, edited):
    edited.transformer.h[0].output[:] = 0
with edited.trace("The Eiffel Tower is in", remote="local"):
    out = edited.transformer.h[0].output.save()   # zeros
```

## Gotchas

- A plain edit applies at the *first* occurrence of a location (prefill, in
  generation); put it under `tracer.iter` to apply at every occurrence.
- `inplace=False` copies the root envoy's edits only. `edited.transformer is
  model.transformer`, so attaching a module, renaming, or storing an edit through a
  *child* of `edited` changes `model` too.
- On a tensor-parallel model, a weight edit needs `.to_local()`: an in-place write
  through a `DTensor` raises, and a reduction over a sharded weight returns this
  rank's answer. See [tensor-parallel](../models/tensor-parallel.md).

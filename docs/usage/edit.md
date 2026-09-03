---
title: Edit
one_liner: Persistently install interventions on a model with `model.edit(inplace=...)`; clear with `model.clear_edits()`.
tags: [usage, edit, persistent]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/usage/skip.md]
sources: [src/nnsight/intervention/editing.py, src/nnsight/intervention/envoy.py]
---

# Edit

## What this is for

`model.edit(...)` opens an editing context whose body — the same intervention DSL as a regular trace — is **captured and stored on the envoy** instead of being executed once. Every subsequent `model.trace(...)` / `model.generate(...)` / `model.pipe(...)` replays the stored interventions (they run *first*, before your invokes).

Two flavors:

- `model.edit()` (default, `inplace=False`) — stores the edit on a **shallow copy** of the envoy; the original is left clean. Entering the block binds `(tracer, edited)`.
- `model.edit(inplace=True)` — stores the edit on the envoy itself. Entering the block binds only `tracer`.

Non-inplace `edit()` yields a `(tracer, edited)` tuple; the `tracer` carries the `iter` API for per-occurrence edits.

## When to use / when not to use

- Use to install always-on transforms (zero a head, add a steering vector, swap in an SAE) without rewriting every trace.
- Use `inplace=True` when every consumer of `model` should see the edit.
- Use the non-inplace form to A/B compare original vs edited.
- Don't use for one-off interventions — that is `model.trace(...)`.

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

## How it works

`Envoy.edit(...)` returns an `EditingTracer` (`src/nnsight/intervention/editing.py`). It captures the with-block like an `InterleavingTracer`, but its `execute` **stores** the captured block as a `Mediator` on `envoy._edits` instead of running it:

```python
self.envoy._edits.append(Mediator(code, globals, dict(locals), copy=True, node=node))
```

On every later trace, `Envoy.interleave` prepends `self._edits` to the run's mediators, so stored interventions run **before** the user's invokes (and an edit's swap is visible to a same-trace read of that location). Non-inplace `edit()` stores on a `_shallow_copy` of the envoy — the underlying `torch.nn.Module`, interleaver, and children are shared, only `_edits` is independent, so no weights are duplicated.

## Multiple edits stack

```python
with model.edit(inplace=True):
    model.transformer.h[0].output[:] = 0     # first edit
with model.edit(inplace=True):
    model.transformer.h[1].output[:] = 0     # second edit — both apply

# len(model._edits) == 2 ; they run in registration order on every trace
```

## Attaching a module in an edit

An edit can attach a module to the tree (adapter/LoRA/SAE) and route activations through it with `hook=True` (runs the module's full `__call__` so its submodules become observable):

```python
model.transformer.h[0].adapter = MyAdapter()
with model.edit(inplace=True):
    acts = model.transformer.h[0].output
    model.transformer.h[0].output[:] = model.transformer.h[0].adapter(acts, hook=True)
```

A plain edit applies once (at the location's first occurrence). To re-apply it at every occurrence — each step of a generation loop — put the passthrough under the tracer's `iter`:

```python
with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:
        acts = model.transformer.h[0].output
        model.transformer.h[0].output[:] = model.transformer.h[0].adapter(acts, hook=True)
```

## Which envoy an edit belongs to

An edit is stored on **the envoy you called `.edit()` on**, and a trace replays
only the edits of the envoy you called `.trace()` on. `Envoy.interleave` sets the
run's mediators from `self._edits`, and `self` is the root of that trace.

That makes a sub-envoy edit a no-op for a root trace, and `clear_edits()` on the
root does not reach it either:

```python
with model.transformer.h[9].edit(inplace=True):
    model.transformer.h[9].output[:] = 0

with model.trace("The Eiffel Tower is in the city of"):
    out = model.output.logits[0, -1].argmax().save()
# ' Paris' — the edit did not run

model.clear_edits()
len(model.transformer.h[9]._edits)      # still 1
```

The edit is real; it runs when that envoy is the root of the trace
(`model.transformer.h[9].trace(hidden_states)` gives zeros), and
`model.transformer.h[9].clear_edits()` removes it. To scope an edit to one layer
of a whole-model run, store it on the model and write the layer's path inside:

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

- Stored edits run **first** on every later trace, before user invokes — their effects are visible to your code.
- Non-inplace `edit()` binds `(tracer, edited)`. Writing the block against `edited` keeps the two models straight in the reader's head; the block lands on the copy either way. `inplace=True` binds only `tracer`.
- A plain edit applies at the *first* occurrence of a location; use the tracer's `iter` to apply at every occurrence.
- An edit stored on a sub-envoy does not run when you trace the root, and the root's `clear_edits()` does not remove it.
- `inplace=False` copies the root envoy's `_edits` only. `edited.transformer is model.transformer`, so attaching a module, renaming, or storing an edit through a *child* of `edited` changes `model` too.
- `model(**ids)` and `model._module(**ids)` bypass the interleaver and run unedited.
- On a tensor-parallel model, a weight edit needs `.to_local()`: an in-place write through a `DTensor` raises, and a reduction over a sharded weight returns this rank's answer. See [../models/tensor-parallel.md](../models/tensor-parallel.md).

## Related

- `docs/usage/trace.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/skip.md`
- `docs/usage/save.md`
- `docs/usage/iter-all-next.md` — re-applying an edit at every generation step

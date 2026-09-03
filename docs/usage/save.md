---
title: Save
one_liner: Persist values from a tracing context with `nnsight.save(obj)` or `obj.save()`. Raises outside a trace.
tags: [usage, tracing, save]
related: [docs/usage/trace.md, docs/usage/scan.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/tracing/tracer.py]
---

# Save

## What this is for

Inside a tracing context (`model.trace`, `model.generate`, `model.pipe`, `model.scan`, `model.session`, `tensor.backward`), variable assignments do **not** automatically escape the with-block. Only values explicitly marked as "saved" are pushed back to the caller's frame. Mark a value with `nnsight.save(value)` or, equivalently, `value.save()`.

## When to use / when not to use

- Use for **every** value you want to read after a `with model.trace(...):` block exits.
- Required inside `model.scan(...)` too — it is a tracing context like the others.
- Not needed for values you only read inside the body.
- **Calling `save` outside a trace raises `ValueError`** (see below).

## Canonical pattern

```python
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello"):
    # Function form: works on any object
    hidden = nnsight.save(model.transformer.h[-1].output)
    # Method form: equivalent
    logits = model.output.logits.save()

print(hidden.shape, logits.shape)
```

## save() raises outside a trace

A save with no trace running can't return anywhere — its mark would be cleared before anything reads it — so it errors instead of silently no-op'ing:

```python
import nnsight
nnsight.save([])
```

```
ValueError: save() was called outside a trace. `.save()` / nnsight.save(x) marks a
value to return from the enclosing `with model.trace(...):` block, so it only works
inside one — move the save into the trace block.
```

The method form raises the same way: `[1, 2, 3].save()` outside a trace → `ValueError: ... outside a trace`.

## How it works

Saving is per-thread state in `src/nnsight/tracing/tracer.py`:

- A trace scope increments a per-thread `depth` around the backend call (`inc`/`dec`).
- `save(value)` raises if `depth == 0`; otherwise it adds `id(value)` to the thread's saved set and returns the value unchanged.
- On exit, `push_result` writes the body's locals back to the caller — but the **outermost** trace (depth 1) keeps only the values whose `id()` is in the saved set. The saved set is cleared when the outermost scope exits.

`save` returns its argument unchanged, so save the value you bind:

```python
h = model.transformer.h[0].output.save()     # h is saved
# NOT: (x.save() * 2) -> saves x, returns the product (unsaved). Write (x * 2).save().
```

## Two forms — prefer `nnsight.save()`

```python
import nnsight

out = nnsight.save(model.transformer.h[0].output)   # function form — recommended
out = model.transformer.h[0].output.save()          # method form
```

The method form relies on a C extension that mounts a `.save()` method onto every Python object at import (check with `hasattr(object(), "save")`). The function form does not, and is unaffected if a class defines its own `.save`. For plain Python types (ints, lists, dicts), always prefer `nnsight.save(...)`.

## Saving non-tensor values

```python
import nnsight

with model.scan("Hello"):
    dim = nnsight.save(model.transformer.h[0].output.shape[-1])   # int
    n_layers = nnsight.save(len(model.transformer.h))             # int

print(dim, n_layers)   # e.g. 768 12
```

## Collecting values in a list (or dict)

To gather values across steps or layers, **save the container and put raw values into it** — do not `.save()` the individual elements:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    per_step = nnsight.save([])                 # save the list itself, inside the trace
    for step in tracer.iter[:5]:
        per_step.append(model.output.logits[0, -1].argmax(dim=-1))   # append raw values
    final = tracer.result.save()
# per_step holds the 5 collected values; final is the generated ids
```

`.save()` marks the object you bind to a name; a saved container comes back with its mutated contents. Two ways this goes wrong:

- **Saving the elements** (`per_step.append(x.save())`) marks values with no name to return them under — it happens to work locally (the list is mutated in your own frame) but returns nothing on a remote trace.
- **Leaving the container unsaved** (`per_step = []` *inside* the trace, or a `[x.save() for ...]` comprehension bound to an unsaved name) never pushes it back, so the name is undefined after the block.

A comprehension follows the same rule — save the whole list, keep elements raw: `hiddens = nnsight.save([b.output for b in model.transformer.h])`.

## Saving inside `tensor.backward()`

The backward context is a nested interleaving session; values saved there reach you (an inner trace pushes everything up, the outermost trace keeps the saved ones):

```python
with model.trace("Hello"):
    a1 = model.transformer.h[0].output
    loss = model.output.logits.sum()
    with loss.backward():
        grad = a1.grad.clone().save()
```

## Remote traces

`.save()` tells the remote backend which values to ship back. Without it, the value is computed on the server and discarded. Move tensors to CPU before saving for smaller transfers:

```python
with model.trace("Hello", remote=True):
    out = model.output.logits.detach().cpu().save()
```

## Gotchas

- Forgetting `.save()` is the most common footgun — the variable is undefined after `__exit__`: `NameError` in a script, `UnboundLocalError` inside a function.
- **A saved value comes back by its variable *name*, so bind it.** `push_result` returns the body's *locals*, filtered to the saved ones — a value marked but never assigned to a name (a bare `model.logits.save()` on its own line) has no local to return, so it silently doesn't come back. This is invisible locally (you just never read it) but obvious on vLLM/remote/serve, where you read it by name: `output.saves["logits"]` will be missing. Always write `logits = model.logits.save()`.
- **Saving outside a trace raises** — do it inside the `with` block.
- `nnsight.save()` is safe to call on the same value multiple times — the saved set is keyed by `id()`.
- Mutating a saved tensor in-place after the trace exits affects whatever it still aliases. Clone for isolation: `nnsight.save(x.clone())`.
- Save is required inside `model.scan(...)` too — it is a tracing context.

## Related

- `docs/usage/trace.md`
- `docs/usage/scan.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/session.md`

---
title: Save
one_liner: Persist values from a tracing context with `nnsight.save(obj)` or `obj.save()`. Raises outside a trace.
tags: [usage, tracing, save]
related: [docs/usage/trace.md, docs/usage/scan.md, docs/usage/access-and-modify.md, docs/usage/session.md]
sources: [src/nnsight/tracing/tracer.py]
---

# Save

Variables assigned inside a tracing context (`model.trace`, `model.generate`, `model.pipe`, `model.scan`, `model.session`, `tensor.backward`) do **not** escape the with-block unless they are marked as saved. Mark every value you want to read after the block with `nnsight.save(value)` or `value.save()`; values you only use inside the body need nothing.

Using an unsaved name after the block raises `NameError: name 'hidden' is not defined` (`UnboundLocalError` inside a function) — the fix is a save inside the block.

## Canonical pattern

```python
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello"):
    hidden = nnsight.save(model.transformer.h[-1].output)   # function form
    logits = model.output.logits.save()                     # method form

print(hidden.shape, logits.shape)
```

`save` returns its argument unchanged, so save the value you bind: `(x.save() * 2)` saves `x` and returns an unsaved product; write `(x * 2).save()`.

**Prefer `nnsight.save()`.** The method form relies on a C extension that mounts `.save()` onto every Python object at import (check with `hasattr(object(), "save")`). The function form does not, and still works when a class defines its own `.save`. For plain Python values (ints, lists, dicts) always use `nnsight.save(...)`. Saving metadata such as shapes inside a scan: see [scan](scan.md).

## save() raises outside a trace

A save with no trace running has nowhere to return to, so it errors instead of silently doing nothing:

```python
import nnsight
nnsight.save([])
```

```
ValueError: save() was called outside a trace. `.save()` / nnsight.save(x) marks a
value to return from the enclosing `with model.trace(...):` block, so it only works
inside one — move the save into the trace block.
```

The method form raises the same way. See [save-outside-trace](../errors/save-outside-trace.md).

## Collecting values in a list (or dict)

To gather values across steps or layers, **save the container and put raw values into it** — do not `.save()` the elements:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    per_step = nnsight.save([])                 # save the list itself, inside the trace
    for step in tracer.iter[:5]:
        per_step.append(model.output.logits[0, -1].argmax(dim=-1))   # append raw values
    final = tracer.result.save()
```

Two ways this goes wrong:

- **Saving the elements** (`per_step.append(x.save())`) marks values with no name to return them under. It happens to work locally (the list is mutated in your own frame) but returns nothing on a remote trace.
- **Leaving the container unsaved** (`per_step = []` inside the trace, or a `[x.save() for ...]` comprehension bound to an unsaved name) never pushes it back, so the name is undefined after the block.

A comprehension follows the same rule: `hiddens = nnsight.save([b.output for b in model.transformer.h])`.

## Saving inside `tensor.backward()`

Values saved inside the backward context reach you like any other save:

```python
with model.trace("Hello"):
    a1 = model.transformer.h[0].output
    loss = model.output.logits.sum()
    with loss.backward():
        grad = a1.grad.clone().save()
```

## Keeping the saves off your variables

A trace's saves reach you only by being injected into the block's variables — the run returns nothing and stores them nowhere else. To keep them on the backend instead, pass a backend built with `inject_results=False`; the saves land on `backend.results`, a dict keyed by the saved names, and the variables stay unbound:

```python
from nnsight.tracing.backend import Backend

backend = Backend(inject_results=False)
with model.trace("Hello", backend=backend):
    logits = nnsight.save(model.lm_head.output)

backend.results["logits"]   # `logits` itself was never bound
```

`CONFIG.APP.INJECT_RESULTS = False` makes it the default for every trace, read from `tracer.backend.results`. It governs only the outermost block: inner traces in a session, and loops over `tracer.iter`, still hand their values on inside the block. Remote runs follow the same setting — see [remote-trace.md](../remote/remote-trace.md#keeping-the-saves-on-the-backend).

## Remote traces

`.save()` tells the remote backend which values to ship back; anything unsaved is computed on the server and discarded. Move tensors to CPU before saving for smaller transfers: `out = model.output.logits.detach().cpu().save()`.

## Gotchas

- **A saved value comes back by its variable *name*, so bind it.** A value marked but never assigned (a bare `model.output.logits.save()` on its own line) has no local to return under, so it silently doesn't come back. Invisible locally, obvious on vLLM/remote/serve where you read it by name: `output.saves["logits"]` will be missing. Write `logits = model.output.logits.save()`.
- **Nested traces don't need `.save()` between them.** Only the outermost boundary filters to saved values; an inner `model.trace()` inside a `model.session()` pushes all its locals up, so a value from one inner trace is usable in the next unsaved (not on `VLLM`, where each trace is its own request). Save only what leaves the session.
- **Unpacking across `.save()` loses both values.** `a, b = x.output.save()` marks the tuple, which is never bound to a name, so `a` and `b` come back unbound. Save first, then unpack: `out = x.output.save()` inside the trace, `a, b = out` after it.
- Saving the same value twice is harmless.
- A saved tensor aliases the live one: writing it in place later in the trace (or after it) changes what you saved. Clone for isolation: `nnsight.save(x.clone())`.

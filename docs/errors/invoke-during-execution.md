---
title: Cannot Invoke While the Model Is Already Running
one_liner: "ValueError: Cannot invoke while the model is already running. — a tracer.invoke(...) was opened inside another invoke's body (or a tracer.iter loop), after the model started executing."
tags: [error, setup, invoke]
related: [docs/errors/cannot-access-outside-interleaving.md, docs/usage/invoke-and-batching.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/tracer.py]
---

# Cannot Invoke While the Model Is Already Running

## Symptom

```
ValueError: Cannot invoke while the model is already running.
```

## Cause

Invokes register their inputs and intervention code **before** the forward pass
starts. Invoke bodies themselves run while the model is executing, so opening a
`tracer.invoke(...)` from inside one has nothing to plug into. Triggers:

- a `tracer.invoke(...)` nested inside another invoke's body;
- a `tracer.invoke(...)` inside a `for step in tracer.iter[...]:` loop.

A `model.trace(...)` opened inside a running trace (directly, or from a helper the
block calls) does not reach this check; it fails with
`AttributeError: 'NoneType' object has no attribute 'event'`. Same fix: don't nest.

## Fix

```python
# WRONG
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        with tracer.invoke("World"):           # ValueError
            out = model.lm_head.output.save()
```

```python
# FIXED — sibling invokes under one trace; or sibling traces under a session
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        a = model.lm_head.output.save()
    with tracer.invoke("World"):
        b = model.lm_head.output.save()

with model.session():
    with model.trace("Hello"):
        a = model.lm_head.output.save()
    with model.trace("World"):
        b = model.lm_head.output.save()
```

To intervene on every generation step, use `tracer.iter[...]` / `tracer.all()`
inside one invoke, not nested invokes.

## Related

- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md)
- [docs/usage/session.md](../usage/session.md)

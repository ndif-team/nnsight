---
title: Cannot Invoke During an Active Model Execution
one_liner: "ValueError: Cannot invoke during an active model execution / interleaving — invoke was opened while the model is already running."
tags: [error, setup, invoker]
related: [docs/errors/model-did-not-execute.md, docs/usage/invoke-and-batching.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/tracing/invoker.py:32]
---

# Cannot Invoke During an Active Model Execution

## Symptom

```
ValueError: Cannot invoke during an active model execution / interleaving.
```

## Cause

`Invoker.__init__` rejects construction whenever the parent tracer's model is already interleaving (`src/nnsight/intervention/tracing/invoker.py:32`):

```python
if tracer is not None and tracer.model.interleaving:
    raise ValueError(
        "Cannot invoke during an active model execution / interleaving."
    )
```

`tracer.model.interleaving` is true between when the interleaver context is entered and when it exits (i.e., while the worker thread is mid-flight). Invokes must all be declared **before** the trace body starts running — they are how the tracer collects batched inputs and registers mediators. Opening a new invoke after the model has already begun executing has no place to plug into.

## Common triggers

- Nesting `tracer.invoke(...)` inside another `tracer.invoke(...)` body.
- Calling `model.trace(...)` from within a function that is itself executing inside another live trace.
- Trying to add a new invoke from inside a `for step in tracer.iter[:]:` loop (the loop runs during interleaving).

## Fix

```python
# WRONG — second invoke is opened inside the first invoke's body
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        with tracer.invoke("World"):       # ValueError
            out = model.lm_head.output.save()
```

```python
# FIXED — sibling invokes, declared sequentially under the same trace
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out_a = model.lm_head.output.save()
    with tracer.invoke("World"):
        out_b = model.lm_head.output.save()
```

```python
# WRONG — calling .trace() from inside a running trace
with model.trace("Hello"):
    with model.trace("World"):             # ValueError on the inner invoker
        out = model.lm_head.output.save()
```

```python
# FIXED — use a session if you need multiple traces sharing state
with model.session() as session:
    with model.trace("Hello"):
        a = model.lm_head.output.save()
    with model.trace("World"):
        b = model.lm_head.output.save()
```

## Mitigation / how to avoid

- Treat invokes as **siblings** under one trace, not children of each other.
- For multiple distinct traces, use `model.session()` (`docs/usage/session.md`).
- For "intervene on every generation step", use `tracer.iter[...]` or `tracer.all()`, not nested invokes.

## Related

- `docs/usage/invoke-and-batching.md`
- `docs/usage/session.md`
- `docs/errors/model-did-not-execute.md`

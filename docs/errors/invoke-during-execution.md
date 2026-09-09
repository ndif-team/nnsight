---
title: Cannot Invoke While the Model Is Already Running
one_liner: "ValueError: Cannot invoke while the model is already running. — a tracer.invoke(...) was opened after the model started executing."
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

`Invoker.__init__` (`src/nnsight/intervention/tracer.py`) rejects construction
when the tracer's interleaver is already interleaving:

```python
if tracer.envoy.interleaver.interleaving:
    raise ValueError("Cannot invoke while the model is already running.")
```

`interleaver.interleaving` is true from when the interleaver context is entered
until it exits — i.e. while the greenlet workers and the model's forward pass are
running. Invokes are how the tracer collects batched inputs and registers workers
**before** the forward pass starts; opening a new one after the model is already
running has nothing to plug into.

## Common triggers

- Nesting `tracer.invoke(...)` inside another `tracer.invoke(...)` body.
- Calling `model.trace(...)` from code that is itself running inside a live trace.
- Opening a new invoke from inside a `for step in tracer.iter[...]:` loop (the loop body runs during interleaving).

## Fix

```python
# WRONG — second invoke opened inside the first invoke's body
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        with tracer.invoke("World"):        # ValueError
            out = model.lm_head.output.save()
```

```python
# FIXED — sibling invokes under the same trace
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        a = model.lm_head.output.save()
    with tracer.invoke("World"):
        b = model.lm_head.output.save()
```

A trace nested inside another trace is a different shape and does not reach this
check — the inner `trace` builds a tracer against an interleaver that is already
running, and fails while wiring it up:

```python
# WRONG — nested traces
with model.trace("Hello"):
    with model.trace("World"):              # AttributeError: 'NoneType' object has no attribute 'event'
        out = model.lm_head.output.save()
```

The fix is the same one.

```python
# FIXED — use a session to run multiple traces
with model.session() as session:
    with model.trace("Hello"):
        a = model.lm_head.output.save()
    with model.trace("World"):
        b = model.lm_head.output.save()
```

## Mitigation

- Treat invokes as **siblings** under one trace, never children of each other.
- For multiple distinct traces, use `model.session()`.
- To intervene on every generation step, use `tracer.iter[...]` / `tracer.all()`, not nested invokes.

## Related

- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md)
- [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md)

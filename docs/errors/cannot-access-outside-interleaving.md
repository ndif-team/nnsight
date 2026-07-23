---
title: Cannot Access Outside of Interleaving
one_liner: "ValueError: Cannot access `<location>` outside of interleaving — an Envoy value was read or set with no trace running; plus: trace() needs an input or an invoke."
tags: [error, setup, interleaving]
related: [docs/errors/save-outside-trace.md, docs/errors/value-was-not-provided.md, docs/usage/trace.md, docs/usage/save.md, docs/concepts/envoy.md]
sources: [src/nnsight/intervention/interleaver.py:225, src/nnsight/intervention/interleaver.py:241, src/nnsight/intervention/tracer.py:266, src/nnsight/intervention/envoy.py:417]
---

# Cannot Access Outside of Interleaving

## Symptom

Reading an Envoy value (`.output`, `.input`, `.inputs`, `.source`, …) with no
trace running:

```
ValueError: Cannot access `model.transformer.h.0.output` outside of interleaving
```

Assigning to one has the **same** message — a swap goes through the same check, so
there is no separate "Cannot set …" text anymore:

```python
model.transformer.h[0].output = value   # ValueError: Cannot access `model.transformer.h.0.output` outside of interleaving
```

A related setup error — a `with model.trace() as tracer:` with no direct input and
no `tracer.invoke(...)` block, so the model has nothing to run on:

```
ValueError: trace() needs an input, or at least one `with tracer.invoke(...)` block
```

> Older docs referenced `ValueError: The model did not execute`. That string does
> **not** exist in this rewrite. The two messages above are what you actually see.

## Cause

Envoy properties resolve through `Mediator.value` / `Mediator.swap`, which call
`Mediator.current` (`src/nnsight/intervention/interleaver.py:225`). That looks for
the greenlet worker driving the current intervention. Intervention code only runs
*while interleaving*, so no worker means the value was requested outside a run —
there is nothing to park on and nothing to answer with, so it raises
`ValueError("Cannot access `<location>` outside of interleaving")`
(`src/nnsight/intervention/interleaver.py:241`). It is raised as a `ValueError`
(not an `AttributeError`) so it isn't swallowed by `__getattr__` and mislabelled as
a missing attribute.

The "trace() needs an input" error comes from `InterleavingTracer.execute`
(`src/nnsight/intervention/tracer.py:266`): if `trace()` got no data input and the
body registered no `tracer.invoke(...)` workers, there is no batch to run.

## Common triggers

- Reading `.output` outside any `with model.trace(...):` block.
- Reading a value **after** the trace block exited without `.save()`-ing it inside.
- A closure that captures an Envoy and runs later, after the trace has exited.
- `with model.trace() as tracer:` with neither a direct input nor any `tracer.invoke(...)`.

## Fix

```python
# WRONG — read happens with no active trace
hidden = model.transformer.h[-1].output   # ValueError: Cannot access ... outside of interleaving
```

```python
# FIXED — read inside a trace, save it, use it after
with model.trace("Hello"):
    hidden = model.transformer.h[-1].output.save()
print(hidden.shape)
```

```python
# WRONG — read after the block exits, never saved
with model.trace("Hello"):
    h = model.transformer.h[0].output       # not saved!
print(h.shape)                              # ValueError: Cannot access ... outside of interleaving
```

```python
# FIXED — call .save() (or nnsight.save(...)) inside the trace
with model.trace("Hello"):
    h = model.transformer.h[0].output.save()
print(h.shape)
```

```python
# WRONG — empty trace with nothing to run
with model.trace() as tracer:               # ValueError: trace() needs an input ...
    out = model.lm_head.output.save()
```

```python
# FIXED — give trace() a direct input, or add an invoke
with model.trace("Hello"):
    out = model.lm_head.output.save()

# or
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out = model.lm_head.output.save()
```

## Related

- [save-outside-trace.md](save-outside-trace.md) — `save()` has its own outside-a-trace guard.
- [value-was-not-provided.md](value-was-not-provided.md) — the model *did* run but the requested module wasn't reached.
- [docs/usage/save.md](../usage/save.md), [docs/usage/trace.md](../usage/trace.md).
- [docs/concepts/envoy.md](../concepts/envoy.md) — how `.output` / `.input` resolve.

---
title: Cannot Access Outside of Interleaving
one_liner: "ValueError: Cannot access `<location>` outside of interleaving — an Envoy value was read or set with no trace running; plus: trace() needs an input or an invoke."
tags: [error, setup, interleaving]
related: [docs/errors/save-outside-trace.md, docs/errors/module-never-fired.md, docs/usage/trace.md, docs/usage/save.md, docs/concepts/envoy.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/tracer.py]
---

# Cannot Access Outside of Interleaving

## Symptom

```
ValueError: Cannot access `model.transformer.h.0.output` outside of interleaving
ValueError: trace() needs an input, or at least one `with tracer.invoke(...)` block
```

## Cause

Envoy values (`.output`, `.input`, `.inputs`, `.source`, …) only exist while a
trace is running the model. Reading or assigning one with no trace running raises
the first message. Common ways to get there: touching `.output` outside any
`with model.trace(...):`, or calling a helper that captured an Envoy after the
trace has exited.

The second message is a `with model.trace() as tracer:` with neither a direct input
nor any `tracer.invoke(...)`, so there is nothing to run.

Reading an unsaved name *after* the trace is a different error: the block's
locals are only bound back if they were saved, so you get `NameError` at module
level or `UnboundLocalError` inside a function. Fix it the same way — `.save()`
inside the block.

## Fix

```python
# WRONG
hidden = model.transformer.h[-1].output        # ValueError: no trace running
with model.trace("Hello"):
    h = model.transformer.h[0].output          # not saved
print(h.shape)                                 # NameError / UnboundLocalError
with model.trace() as tracer:                  # ValueError: trace() needs an input
    pass
```

```python
# FIXED — read inside a trace (with an input), save what you need afterwards
with model.trace("Hello"):
    hidden = model.transformer.h[-1].output.save()
print(hidden.shape)
```

## Related

- [save-outside-trace.md](save-outside-trace.md) — `save()` has its own outside-a-trace guard.
- [module-never-fired.md](module-never-fired.md) — the model ran but the module wasn't reached.
- [docs/usage/save.md](../usage/save.md), [docs/usage/trace.md](../usage/trace.md).

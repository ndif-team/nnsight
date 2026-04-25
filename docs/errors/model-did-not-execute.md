---
title: Cannot Access Outside of Interleaving
one_liner: "ValueError: Cannot access `<path>.output` outside of interleaving — Envoy property accessed when no trace is running."
tags: [error, setup, interleaving]
related: [docs/errors/missed-provider-error.md, docs/usage/trace.md, docs/usage/save.md, docs/concepts/envoy-and-eproperty.md]
sources: [src/nnsight/intervention/interleaver.py:302, src/nnsight/intervention/interleaver.py:326, src/nnsight/intervention/envoy.py:47]
---

# Cannot Access Outside of Interleaving

## Symptom

The exact text emitted by the `eproperty` descriptor for `.output` / `.input` / `.inputs` / `.source` etc.:

```
ValueError: Cannot access `model.transformer.h.0.output` outside of interleaving.
```

Or, for assignment (`module.output = value`):

```
ValueError: Cannot set `model.transformer.h.0.output` outside of interleaving.
```

Or, for trace-only methods (`.skip`, `.next`, etc.) called with no live trace:

```
ValueError: Must be within a trace to use `.skip(...)`
```

> Older docs (and `CLAUDE.md`) reference a `ValueError: The model did not execute` — that exact string does not appear in the current source. The actual symptom users see is one of the three above.

## Cause

The `eproperty` descriptor backing `.output` / `.input` / `.inputs` checks `interleaver.interleaving` before serving a value (`src/nnsight/intervention/interleaver.py:271`). When that flag is false, accessing the property raises immediately — there is no model running, so the Envoy has nothing to give.

Concretely the flag is true only while:

- A trace body is executing inside `Envoy.interleave` (`src/nnsight/intervention/envoy.py:590`), and
- The interleaver context is still entered (i.e., before `__exit__`).

The `trace_only` decorator (`src/nnsight/intervention/envoy.py:41`) provides the analogous error for methods like `.skip()`, `.next()`.

## Common triggers

- Reading `.output` outside any `with model.trace(...):` block.
- Assigning to `.output` outside a trace.
- Reading `.output` after the trace's `with` block has already exited (the value was never `.save()`'d, so the proxy is stale).
- Calling `model.trace()` with no positional input AND no `tracer.invoke(input)` body — the model never runs, so any `.output` access inside fires this error rather than something more descriptive.
- Closures defined inside loops that capture an Envoy and run them later, after the trace has exited.

## Fix

```python
# WRONG — read happens outside any active trace
hidden = model.transformer.h[-1].output  # ValueError: Cannot access ... outside of interleaving.
```

```python
# FIXED — read inside a trace, save the value, use it after
with model.trace("Hello"):
    hidden = model.transformer.h[-1].output.save()
print(hidden.shape)
```

```python
# WRONG — empty trace with nothing to run
with model.trace() as tracer:
    out = model.lm_head.output.save()  # model never actually runs
```

```python
# FIXED — provide input on .trace(...) or via tracer.invoke(...)
with model.trace("Hello"):
    out = model.lm_head.output.save()

# or
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out = model.lm_head.output.save()
```

```python
# WRONG — accessing after the trace exits
with model.trace("Hello"):
    h = model.transformer.h[0].output      # not saved!
print(h.shape)                              # ValueError: Cannot access ... outside of interleaving.
```

```python
# FIXED — call .save() (or nnsight.save(...)) inside the trace
with model.trace("Hello"):
    h = model.transformer.h[0].output.save()
print(h.shape)
```

## Mitigation / how to avoid

- Always pair a trace with an input — either positional on `.trace(input)` or in an explicit `tracer.invoke(input)`.
- Always `.save()` (or `nnsight.save(...)`) any value you want to read after the `with` block exits; see `docs/usage/save.md`.
- If you need to debug from a closure, read the value inside the trace and bind it to a saved variable.

## Related

- `docs/errors/missed-provider-error.md` (the failure mode if the model does run but the requested module isn't called)
- `docs/usage/trace.md`
- `docs/usage/save.md`
- `docs/concepts/envoy-and-eproperty.md`

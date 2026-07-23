---
title: save() Called Outside a Trace
one_liner: "ValueError: save() was called outside a trace — .save() / nnsight.save(x) only marks a value inside a `with model.trace(...):` block."
tags: [error, setup, save]
related: [docs/errors/cannot-access-outside-interleaving.md, docs/usage/save.md, docs/usage/trace.md]
sources: [src/nnsight/tracing/tracer.py:161, src/nnsight/tracing/tracer.py:174]
---

# save() Called Outside a Trace

## Symptom

```
ValueError: save() was called outside a trace. `.save()` / nnsight.save(x) marks a value to return from the enclosing `with model.trace(...):` block, so it only works inside one — move the save into the trace block.
```

Both `nnsight.save(x)` and the mounted `x.save()` form raise this when no trace is
running.

## Cause

`save` (`src/nnsight/tracing/tracer.py:161`) marks a value by identity to be
returned once the outermost trace exits. That only means something inside a trace —
a saved value is what the `with model.trace(...):` block hands back. Calling it with
no trace running would mark into a saved set that is cleared before anything reads
it, so instead of a silent no-op it is now an explicit error
(`src/nnsight/tracing/tracer.py:174`, guarded on the per-thread trace depth being 0).

> This is a behavior change from older nnsight, where `save()` outside a trace was
> a silent no-op.

## Fix

```python
import nnsight

# WRONG — saved before the block; nothing is tracing yet
acts = nnsight.save([])          # ValueError: save() was called outside a trace
with model.trace("Hello"):
    acts.append(model.transformer.h[0].output)
```

```python
# FIXED — save the value you bind, inside the trace
with model.trace("Hello"):
    acts = model.transformer.h[0].output.save()
print(acts.shape)
```

Save the concrete object you want back — `save` returns its argument unchanged, so
save the value you bind:

```python
with model.trace("Hello"):
    doubled = (model.transformer.h[0].output * 2).save()   # save the product...
    # NOT:  x.save() * 2  — that returns x, and only x is saved
```

## Note for internal callers

`nnsight.tracing.tracer.mark(value)` is the same mechanism without the guard, for
code that must mark values to return *outside* a running trace (e.g. a remote
backend recording a finished request's returned values). User code should use
`save`.

## Related

- [docs/usage/save.md](../usage/save.md) — the full `save` semantics.
- [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) — reading an Envoy value outside a trace.

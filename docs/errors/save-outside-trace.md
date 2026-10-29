---
title: save() Called Outside a Trace
one_liner: "ValueError: save() was called outside a trace — .save() / nnsight.save(x) only marks a value inside a `with model.trace(...):` block."
tags: [error, setup, save]
related: [docs/errors/cannot-access-outside-interleaving.md, docs/usage/save.md, docs/usage/trace.md]
sources: [src/nnsight/tracing/tracer.py]
---

# save() Called Outside a Trace

## Symptom

```
ValueError: save() was called outside a trace. `.save()` / nnsight.save(x) marks a value to return from the enclosing `with model.trace(...):` block, so it only works inside one — move the save into the trace block.
```

## Cause

`save` marks a value to be handed back when the trace block exits, so it only means
something inside one. Both `nnsight.save(x)` and `x.save()` raise this when no trace
is running.

## Fix

```python
# WRONG — saved before the block
acts = nnsight.save([])                        # ValueError
with model.trace("Hello"):
    acts.append(model.transformer.h[0].output)

# FIXED — save the value you bind, inside the trace
with model.trace("Hello"):
    acts = model.transformer.h[0].output.save()
    doubled = (model.transformer.h[0].output * 2).save()   # not x.save() * 2 — that saves x
```

## Marking is by identity, not by name

`save` records the object, and the block returns every local bound to a marked
object. For a value Python interns — a small int, `True`, `None`, a short string —
any *other* local in the block holding the same object comes back too:

```python
with model.trace("Hello"):
    kept = nnsight.save(1)
    unrelated = 1          # same interned object; also bound after the block
```

Harmless, but it explains a name you never saved being defined after the block.
Conversely, a `.save()` whose result is never bound to a name does not come back.

## Related

- [docs/usage/save.md](../usage/save.md) — the full `save` semantics.
- [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) — reading an Envoy value outside a trace.

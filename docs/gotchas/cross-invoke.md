---
title: Cross-Invoke Pitfalls
one_liner: A name that already exists outside the invokes shadows the one a sibling invoke binds, silently.
tags: [gotcha, invoke, barrier]
related: [docs/usage/invoke-and-batching.md, docs/usage/barrier.md, docs/concepts/batching-and-invokers.md]
sources: [src/nnsight/tracing/util.py, src/nnsight/intervention/barrier.py]
---

# Cross-Invoke Pitfalls

Whether a name one invoke binds is readable in another comes down to where each
worker has parked. That rule, the park-past pattern it allows, and the barrier
that covers the rest are in
[usage/invoke-and-batching.md](../usage/invoke-and-batching.md#cross-invoke-value-sharing).
Everything there fails loudly, with a `NameError` naming the variable.

This page is for the one that does not.

## A name from the enclosing scope shadows the one an invoke binds

### Symptom

No error. The consuming invoke runs, uses the value it expected to receive, and
the numbers are wrong — usually zeros, or a previous experiment's tensor.

### Cause

Each block starts from a copy of the surrounding frame's locals and reads that
copy first; a bind from a sibling invoke reaches it through the shared dict
(`Scope.shared` in `tracing/util.py`) only when the name is not already in the
copy. So a `donor` that exists in the notebook or the enclosing function is the
one the consumer sees, whatever the producer binds later.

### Wrong

```python
donor = torch.zeros(768)          # left over from an earlier cell

with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        donor = model.transformer.h[5].output[:, -1].clone()
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[:, -1] = donor    # the zeros, not the clone
        logits = model.output.logits.save()
```

### Right

Name it something that exists nowhere else:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        eiffel_h5 = model.transformer.h[5].output[:, -1].clone()
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[:, -1] = eiffel_h5
        logits = model.output.logits.save()
```

Run both and the patched position has norm `0.0` in the first and `77.6` in the
second: the write landed, it just landed the wrong tensor.

### How to spot it early

A cross-invoke handoff that produces no error *and* no effect is this, not a
barrier problem — a missing barrier raises. In a notebook, check whether the
donor name survives from an earlier cell.

## Related

- [usage/invoke-and-batching.md](../usage/invoke-and-batching.md) — when a
  cross-invoke name is readable, and the barrier for when it is not.
- [usage/barrier.md](../usage/barrier.md) — barrier reference.
- [gotchas/modification.md](modification.md) — `.clone()` for a slice shared across
  invokes.

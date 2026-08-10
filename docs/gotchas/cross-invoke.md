---
title: Cross-Invoke Pitfalls
one_liner: Sharing values between invokes — when a barrier(n) is required and how empty invokes behave on bare NNsight.
tags: [gotcha, invoke, barrier]
related: [docs/usage/invoke-and-batching.md, docs/usage/barrier.md, docs/concepts/batching-and-invokers.md]
sources: [src/nnsight/intervention/barrier.py, src/nnsight/tracing/util.py:32, src/nnsight/intervention/envoy.py:597]
---

# Cross-Invoke Pitfalls

## TL;DR
- A value **defined before/around** the invokes (in the enclosing scope) flows into every invoke automatically — no barrier.
- A value **produced inside one invoke** (captured from an activation) is **not** automatically visible to a sibling invoke: all invoke workers start together, so the consumer runs before the producer has bound it → `NameError`. Use `tracer.barrier(n)` to order them.
- This applies whether the invokes touch the **same** module or **different** modules — the old "different modules → no barrier needed" rule no longer holds.
- **Use a name that doesn't already exist outside the invokes.** Each block starts with a copy of the surrounding scope, and that copy is checked first — so if `donor` already exists out there, the consumer reads the *old* value instead of the one the producer just bound. Nothing errors.
- `CONFIG.APP.CROSS_INVOKER` is **gone**. Sharing is via the blocks' shared frame locals (`Scope.shared`), sequenced by barriers.
- Empty `tracer.invoke()` (no input) works on bare `NNsight` — it contributes no rows and skips `_batch()`, so it never raises `NotImplementedError`.

---

## Passing a captured value between invokes needs a barrier

### Symptom
```
NameError: name 'clean_hs' is not defined
```
when invoke 2 uses a variable that invoke 1 captured from a module.

### Cause
Each invoke's body runs in its own **greenlet worker**, and all workers start (run to their first park) as soon as the interleaver is entered. Blocks written in the same frame share their local binds through a shared dict (`Scope.shared`, `src/nnsight/tracing/util.py:32`) — but only *after* the producing worker has actually executed the assignment. Invoke 1 parks on its `.output` read before binding `clean_hs`; invoke 2 starts and immediately references `clean_hs`, which isn't bound yet → `NameError`.

`tracer.barrier(n)` (`src/nnsight/intervention/barrier.py`) forces the ordering: invoke 1 binds the value, then calls `barrier()`; invoke 2's first action is `barrier()`, so it waits until invoke 1 has arrived — by which point the value is bound.

### Wrong code
```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[:, -1, :].clone()
    with tracer.invoke("The Colosseum is in"):
        model.transformer.h[5].output[:, -1, :] = clean_hs   # NameError
        logits = model.output.logits.save()
```

### Right code (with barrier)
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[:, -1, :].clone()
        barrier()       # invoke 1 has captured clean_hs
    with tracer.invoke("The Colosseum is in"):
        barrier()       # invoke 2 waits here until invoke 1 arrives
        model.transformer.h[5].output[:, -1, :] = clean_hs
        logits = model.output.logits.save()
```

### Mitigation / how to spot it early
- `NameError` for a cross-invoke variable → the producer hadn't bound it yet. Add `tracer.barrier(2)`, call it after the capture in the producer and before the use in the consumer.
- `.clone()` the captured slice (see [modification.md](modification.md)) so it survives the second invoke's writes.

---

## When you don't need a barrier

### Cause
A value bound **before** the invokes (or anywhere outside them) is captured in each block's scope snapshot, so every invoke sees it with no synchronization.

### Right code (no barrier)
```python
steer = torch.zeros(768)                       # defined in the enclosing scope
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        a = model.transformer.h[0].output[:, -1, :].save()
    with tracer.invoke("World"):
        model.transformer.h[5].output[:, -1, :] += steer   # outer value — fine
        b = model.output.logits.save()
```

### Mitigation
- Ask "was this value produced *inside* another invoke?" If yes → barrier. If it came from the enclosing scope → no barrier.

---

## Empty `tracer.invoke()` on bare `NNsight`

### Symptom
On a base `NNsight` model, two input invokes raise:
```
NotImplementedError: NNsight does not support batching multiple invokes
```

### Cause
Batching two or more inputs requires `_batch()` (`src/nnsight/intervention/envoy.py:597`), which base `NNsight` doesn't implement — only batching models like `TransformersModel` do. An *empty* `tracer.invoke()` contributes no rows and skips `_batch()`, spawning a worker over the existing batch.

### Wrong / Right (bare NNsight)
```python
import torch
from nnsight import NNsight

model = NNsight(torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 2)))

# wrong — two input invokes
with model.trace() as tracer:
    with tracer.invoke(torch.rand(1, 5)):
        a = model[0].output.save()
    with tracer.invoke(torch.rand(1, 5)):   # NotImplementedError
        b = model[0].output.save()

# right — one input + empty invoke(s)
with model.trace() as tracer:
    with tracer.invoke(torch.rand(1, 5)):
        a = model[0].output.save()
    with tracer.invoke():                   # same batch, fresh worker
        b = model[1].output.save()
```

### Mitigation / how to spot it early
- For bare `NNsight`, use *one* input invoke plus empty invokes. For multiple input invokes, use a batching model or implement `_batch_size`/`_batch`.

---

## Related
- [docs/usage/barrier.md](../usage/barrier.md) — barrier reference.
- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md) — invokes and batching.
- [docs/gotchas/modification.md](modification.md) — `.clone()` for slices shared across invokes.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — within-invoke ordering.

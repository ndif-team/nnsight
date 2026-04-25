---
title: Cross-Invoke Pitfalls
one_liner: Sharing values between invokes — when you need a barrier(), what CROSS_INVOKER does, and how empty invokes behave on bare NNsight.
tags: [gotcha, invoke, barrier, cross-invoker]
related: [docs/usage/invoke-and-batching.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/tracing/tracer.py:551, src/nnsight/intervention/interleaver.py:1123, src/nnsight/intervention/interleaver.py:889, src/nnsight/intervention/interleaver.py:1207]
---

# Cross-Invoke Pitfalls

## TL;DR
- Variables defined in one invoke flow into later invokes by default — but only when no module access in the second invoke would have already overtaken the value's materialization point.
- If two invokes both access the *same module*, sharing a Python variable between them needs `tracer.barrier(n)` to synchronize the two threads at that module.
- `CONFIG.APP.CROSS_INVOKER` (default `True`) is what makes the cross-invoke variable flow happen at all. Setting it to `False` isolates each invoke (sometimes useful for debugging).
- Empty `tracer.invoke()` (no positional args) operates on the entire batch from earlier invokes. It works on bare `NNsight` models because it does not call `_batch()`.
- Decision rule: same module accessed in both invokes? → barrier. Different modules? → no barrier needed.

---

## Same-module cross-invoke share without a barrier

### Symptom
`NameError: name 'clean_hs' is not defined` (or similar) when invoke 2 tries to use a variable defined in invoke 1, and both invokes touch the same module.

### Cause
Invokes run as serial worker threads. Cross-invoke variable propagation works via the `Mediator.send` push/pull mechanism (`src/nnsight/intervention/interleaver.py:1207`): variables from one invoke's frame are pushed to a shared frame, and the next invoke pulls from there. The push happens *at every event* (request/swap), so a variable becomes visible to other invokes only once invoke 1 has reached an event.

When both invokes access the same module path (say `transformer.h[5].output`):
- Invoke 1 reaches its `.output` access first, gets the value, assigns it to `clean_hs`.
- Invoke 2 reaches *its* `.output` access — but the model's hook already fired and was consumed by invoke 1's request. The value is past.

Without explicit synchronization, invoke 2 sees nothing for that path. The fix is `tracer.barrier(n)`, which forces invoke 1 to wait until it has materialized the variable, *then* lets invoke 2 proceed.

### Wrong code
```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :].clone()
    with tracer.invoke("The Colosseum is in"):
        model.transformer.h[5].output[0][:, -1, :] = clean_hs   # NameError
        logits = model.lm_head.output.save()
```

### Right code (with barrier)
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :].clone()
        barrier()       # invoke 1 reaches barrier after capturing clean_hs
    with tracer.invoke("The Colosseum is in"):
        barrier()       # invoke 2 waits here until invoke 1 has reached its barrier
        model.transformer.h[5].output[0][:, -1, :] = clean_hs
        logits = model.lm_head.output.save()
```

### Mitigation / how to spot it early
- Decision rule: if two invokes both access `.output`/`.input` on the same module and you want to share a value between them, use a barrier.
- For different modules (invoke 1 reads `h[3]`, invoke 2 reads `h[7]`), no barrier is needed — invoke 1 finishes before invoke 2 reaches `h[7]`.

See `Barrier` in `src/nnsight/intervention/tracing/tracer.py:646` and `handle_barrier_event` in `src/nnsight/intervention/interleaver.py:1123` for the implementation.

---

## When you don't need a barrier

### Symptom
You add a barrier "just in case" and notice the trace works fine without it. Or you're worried about a `NameError` and add unnecessary synchronization.

### Cause
Cross-invoke variable sharing works *automatically* via the `cross_invoker` push/pull mechanism (`src/nnsight/intervention/interleaver.py:889`) when the invokes don't compete on the same module access. As long as invoke 1's variable is captured *before* invoke 2 reaches its first `.output`/`.input` access (which is true when invoke 2 reads a *different* module), the variable propagates.

### Right code (no barrier)
```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        h2 = model.transformer.h[2].output[0].clone().save()    # only invoke 1 reads h[2]
    with tracer.invoke("World"):
        # only invoke 2 reads h[5] — no module conflict, no barrier needed
        model.transformer.h[5].output[0][:] = h2
        logits = model.lm_head.output.save()
```

### Mitigation / how to spot it early
- The check is "do both invokes access the same Envoy?" If no, skip the barrier.
- Adding an unneeded barrier doesn't hurt correctness but adds a synchronization step.

---

## `CROSS_INVOKER = False` — fully isolated invokes

### Symptom
After setting `CONFIG.APP.CROSS_INVOKER = False`, every cross-invoke variable reference raises `NameError`, even with a barrier.

### Cause
`Mediator.start` reads the config flag once: `cross_invoker = len(self.interleaver.mediators) > 1 and CONFIG.APP.CROSS_INVOKER` (`src/nnsight/intervention/interleaver.py:889`). When false, `Mediator.send` skips the `push()`/`pull()` calls entirely, so each invoke's frame stays isolated.

This is mostly useful for debugging — it forces you to be explicit about every value you transfer (for instance, by saving and reloading inside both invokes).

### Wrong assumption
```python
nnsight.CONFIG.APP.CROSS_INVOKER = False

with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("A"):
        x = model.transformer.h[2].output[0].clone()
        barrier()
    with tracer.invoke("B"):
        barrier()
        model.transformer.h[5].output[0][:] = x   # NameError — frame is isolated
```

### Right approach
Either re-enable `CROSS_INVOKER`, or save the value and pass it through Python state:

```python
# either re-enable
nnsight.CONFIG.APP.CROSS_INVOKER = True

# or do the work in two passes (different traces)
with model.trace("A"):
    saved_x = model.transformer.h[2].output[0].clone().save()

with model.trace("B"):
    model.transformer.h[5].output[0][:] = saved_x
    logits = model.lm_head.output.save()
```

### Mitigation / how to spot it early
- If you see `NameError` for cross-invoke variables and have set `CROSS_INVOKER = False`, that's why.

---

## Empty `tracer.invoke()` on bare `NNsight`

### Symptom
On a base `NNsight` model (not `LanguageModel`), passing multiple inputs as separate invokes raises `NotImplementedError: Batching is not implemented`. But you want to access the same input multiple times in different blocks.

### Cause
`Batcher.batch(...)` requires `_prepare_input()` and `_batch()` methods on the model class. Base `NNsight`/`Envoy` doesn't implement them — only `LanguageModel` (and similar specialized wrappers) does. Passing inputs to multiple invokes therefore needs batching machinery that may not exist.

But: an *empty* invoke (`tracer.invoke()` with no args) does not contribute new inputs to the batch. It just spawns a worker thread that operates on the existing batch from previous invokes. This bypasses `_batch()` entirely.

So: one input invoke + any number of empty invokes works on bare `NNsight`. Multiple input invokes does not.

### Wrong code (bare NNsight)
```python
import torch
from nnsight import NNsight

net = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 2))
model = NNsight(net)

with model.trace() as tracer:
    with tracer.invoke(torch.rand(1, 5)):
        a = model[0].output.save()
    with tracer.invoke(torch.rand(1, 5)):    # NotImplementedError
        b = model[0].output.save()
```

### Right code (one input + empty invokes)
```python
with model.trace() as tracer:
    with tracer.invoke(torch.rand(1, 5)):
        a = model[0].output.save()
    with tracer.invoke():                    # empty — works on the same batch
        b = model[1].output.save()
```

### Mitigation / how to spot it early
- For bare `NNsight`, restrict yourself to *one* input invoke per trace.
- Use empty invokes to access modules in different orders (each is its own thread, so the forward-pass order rule resets).
- For models that need multiple input invokes, use `LanguageModel` (which implements `_prepare_input`/`_batch`) or implement those methods on your subclass.

---

## Related
- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md) — invoke / barrier reference.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — full thread/mediator architecture.
- [docs/gotchas/modification.md](modification.md) — `.clone()` is often required when sharing slices across invokes.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — within-invoke ordering rules.

---
title: Ordering and Deadlock Pitfalls
one_liner: Rules about WHEN you can access modules — out-of-order access, missing input, and nested invokes.
tags: [gotcha, order, deadlock, threading]
related: [docs/concepts/threading-and-mediators.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/interleaver.py:1013, src/nnsight/intervention/tracing/invoker.py:32, src/nnsight/intervention/interleaver.py:652]
---

# Ordering and Deadlock Pitfalls

## TL;DR
- Within a single invoke, request modules in *forward-pass order*. Asking for `layer 5` then `layer 2` raises `OutOfOrderError` (the worker thread is still waiting for layer 2's value but layer 2 already ran).
- `model.trace()` with no positional input *and* no `tracer.invoke(...)` inside it raises — there's nothing to feed the model.
- You cannot create a `tracer.invoke(...)` *inside* an active forward pass. Invokes are top-level; nesting them raises `ValueError: Cannot invoke during an active model execution / interleaving.`

---

## Out-of-order module access (deadlock / OutOfOrderError)

### Symptom
- `Mediator.OutOfOrderError: Value was missed for model.transformer.h.2.output.i0. Did you call an Envoy out of order?` (the actual exception class is in `src/nnsight/intervention/interleaver.py:760`).
- Or, in some cases, the trace finishes with `ValueError: Execution complete but '...' was not provided`.

### Cause
Each invoke runs in a worker thread that issues blocking `request(...)` calls for `.output`/`.input` and waits until the model's forward pass calls a one-shot hook with that value. The worker steps through the model in linear order: when it requests layer 5's output, the mediator's history records that layer 2 has already been seen and *passed*, so a later request for layer 2 is recognized as out-of-order and rejected (see `Mediator.handle_value_event` in `src/nnsight/intervention/interleaver.py:1013`).

Code in your invoke runs *as Python*, but each `.output` access is a synchronization point with the model. You can't go backwards.

### Wrong code
```python
with model.trace("Hello"):
    out5 = model.transformer.h[5].output.save()   # waits, fires at layer 5
    out2 = model.transformer.h[2].output.save()   # OutOfOrderError — layer 2 is already past
```

### Right code (single invoke, in order)
```python
with model.trace("Hello"):
    out2 = model.transformer.h[2].output.save()
    out5 = model.transformer.h[5].output.save()
```

### Right code (two passes via separate invokes)
If you genuinely need them collected in the wrong order from a logical standpoint, use two invokes — each is a separate forward pass, so each thread starts fresh:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out5 = model.transformer.h[5].output.save()
    with tracer.invoke():    # empty invoke — operates on the same batch
        out2 = model.transformer.h[2].output.save()
```

### Mitigation / how to spot it early
- If you see `OutOfOrderError`, match the path in the message against your code and confirm you accessed it after a later module.
- The error message's `.i0` suffix is the iteration counter — `i0` is the first forward pass.

---

## `model.trace()` with no input and no invokes

### Symptom
The trace block exits, then somewhere in the post-processing you get `ValueError: Execution complete but 'lm_head.output.i0' was not provided`, or simply: the model never ran. In some cases nothing at all happens — the saved variables are empty / never touched.

### Cause
`.trace(*args, **kwargs)` with positional args creates an implicit invoke for those args. With no positional args and no inner `tracer.invoke(...)` calls, the batcher has zero batched inputs and the model is never called. The interleaver still runs the worker threads, which then block forever waiting for values that will never arrive — at exit time `check_dangling_mediators` (`src/nnsight/intervention/interleaver.py:652`) flags this as a missed provider.

### Wrong code
```python
with model.trace():
    output = model.lm_head.output.save()
```

### Right code
```python
# implicit invoke
with model.trace("Hello"):
    output = model.lm_head.output.save()

# or explicit
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        output = model.lm_head.output.save()
```

### Mitigation / how to spot it early
- The smell is "trace context with no positional args and no `tracer.invoke(...)` block".
- If you see a `MissedProviderError` for the *first* module the model would have called, this is likely the cause.

---

## Nested / mid-execution `tracer.invoke(...)`

### Symptom
`ValueError: Cannot invoke during an active model execution / interleaving.` This raises immediately when you enter a nested `tracer.invoke(...)`.

### Cause
`Invoker.__init__` checks `tracer.model.interleaving` and refuses if the model is currently executing (see `src/nnsight/intervention/tracing/invoker.py:32`). Each invoke's mediator is created and registered *before* the model starts running; once the forward pass is in flight, you cannot add new mediators to it.

This commonly happens when:
- Someone tries to nest `with tracer.invoke(...)` blocks inside each other.
- Code defines an invoke inside a generator iterator step (`for step in tracer.iter[:]:` body).
- Code defines an invoke inside another worker thread's intervention.

### Wrong code
```python
# nested invokes
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        with tracer.invoke("World"):    # ValueError
            ...

# invoke inside an iter step
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[:]:
        with tracer.invoke("X"):    # ValueError — model is interleaving
            ...
```

### Right code
```python
# sequential, top-level invokes
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        ...
    with tracer.invoke("World"):
        ...
```

### Mitigation / how to spot it early
- Invokes are always direct children of a `tracer` that has *not* started executing yet. Never nest them.
- If you need cross-invoke synchronization within a step, use `tracer.barrier(n)` instead of trying to nest.

---

## Related
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — full mental model of the mediator threads and event queues.
- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md) — how invokes batch inputs.
- [docs/gotchas/iteration.md](iteration.md) — the unbounded-iter footgun closely interacts with module-call ordering.
- [docs/gotchas/cross-invoke.md](cross-invoke.md) — barriers when two invokes both touch the same module.

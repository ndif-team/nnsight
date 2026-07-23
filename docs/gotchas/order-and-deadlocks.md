---
title: Ordering and Execution Pitfalls
one_liner: Rules about WHEN you can access modules — out-of-order access (OutOfOrderError, not a hang), missing input, and nested invokes.
tags: [gotcha, order, execution, greenlets]
related: [docs/concepts/threading-and-mediators.md, docs/errors/out-of-order-error.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/interleaver.py:83, src/nnsight/intervention/interleaver.py:605, src/nnsight/intervention/tracer.py:266, src/nnsight/intervention/tracer.py:346]
---

# Ordering and Execution Pitfalls

## TL;DR
- Within a single block, request modules in **forward-pass order**. Asking for layer 5's output then layer 2's raises **`OutOfOrderError`** — not a deadlock/hang. Interleaving is cooperative (greenlets), so an unservable request surfaces as an error when the run finishes, with the traceback on the waiting line.
- `model.trace()` with no positional input *and* no `tracer.invoke(...)` raises `ValueError: trace() needs an input, or at least one \`with tracer.invoke(...)\` block`.
- You cannot open a `tracer.invoke(...)` while the model is running (nested invokes, or an invoke inside an `iter` loop) — `ValueError: Cannot invoke while the model is already running.`

---

## Out-of-order module access (`OutOfOrderError`)

### Symptom
```
nnsight.intervention.interleaver.OutOfOrderError: 'model.transformer.h.2.output.i0' was requested but the model already ran past it
```
Import to catch it: `from nnsight.intervention.interleaver import OutOfOrderError`.

### Cause
Each block runs in its own **greenlet worker** (a `Mediator`) that runs in lockstep with the forward pass. Reading `module.output` *parks* the worker until the model reaches that module. A worker holds one pending request at a time and can only be served locations in the order the model reaches them. If the run finishes with a worker still parked on a location the model already ran past (or never reached), `check_dangling_mediators` (`src/nnsight/intervention/interleaver.py:605`) throws `OutOfOrderError` into the worker so the traceback points at the exact waiting line. The `.i0` suffix is the occurrence index (`i0` = first forward pass; see [iteration.md](iteration.md)).

### Wrong code
```python
with model.trace("Hello"):
    out5 = model.transformer.h[5].output.save()
    out2 = model.transformer.h[2].output.save()   # OutOfOrderError — layer 2 already ran
```

### Right code (single block, in order)
```python
with model.trace("Hello"):
    out2 = model.transformer.h[2].output.save()
    out5 = model.transformer.h[5].output.save()
```

### Right code (two passes via separate invokes)
Each invoke is its own worker, so its ordering resets:
```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out5 = model.transformer.h[5].output.save()
    with tracer.invoke():                 # empty invoke — same batch, fresh worker
        out2 = model.transformer.h[2].output.save()
```

### Mitigation / how to spot it early
- Read the location in the message and confirm you accessed it after a later module.
- Note attention runs *inside* its block: reading `h[0].output` then `h[0].attn.output` is out of order (attn already ran). Read the finer submodule first.

---

## `model.trace()` with no input and no invokes

### Symptom
- `ValueError: trace() needs an input, or at least one \`with tracer.invoke(...)\` block` (body has no module access), or
- `ValueError: Cannot access \`model.output\` outside of interleaving` (body reads a module — the block runs to collect invokes *before* the model starts, so no worker exists yet).

### Cause
`trace(*args)` with positional args creates one implicit invoke. With no args and no inner `tracer.invoke(...)`, there is no batched input; `execute` (`src/nnsight/intervention/tracer.py:266`) raises rather than run the model on nothing.

### Wrong / Right
```python
# wrong
with model.trace():
    output = model.output.logits.save()

# right — implicit invoke
with model.trace("Hello"):
    output = model.output.logits.save()

# right — explicit invoke
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        output = model.output.logits.save()
```

---

## Nested / mid-execution `tracer.invoke(...)`

### Symptom
```
ValueError: Cannot invoke while the model is already running.
```

### Cause
`Invoker.__init__` (`src/nnsight/intervention/tracer.py:346`) refuses if the interleaver is already interleaving. Invokes are collected *before* the model runs; once a worker is executing (inside another invoke's body, or an `iter` step), you can't register a new one.

### Wrong code
```python
# nested invokes
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        with tracer.invoke("World"):    # ValueError
            ...

# invoke inside an iter step
with model.generate("Hello", max_new_tokens=3) as tracer:
    for _ in tracer.iter[:3]:
        with tracer.invoke("X"):        # ValueError
            ...
```

### Right code
```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        ...
    with tracer.invoke("World"):
        ...
```

### Mitigation / how to spot it early
- Invokes are always direct, sequential children of a `tracer` that hasn't started running. For cross-invoke synchronization, use `tracer.barrier(n)` (see [cross-invoke.md](cross-invoke.md)).

---

## Related
- [docs/errors/out-of-order-error.md](../errors/out-of-order-error.md) — the error in detail.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — greenlet workers and the interleaver.
- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md) — how invokes batch inputs.
- [docs/gotchas/iteration.md](iteration.md) — ordering across generation steps.

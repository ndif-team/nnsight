---
title: Threading and Mediators
one_liner: Each block of intervention code is one Mediator running in a greenlet (not a thread) that parks on a location and switches back to the model, exchanging one typed event (VALUE, SWAP, SKIP, BARRIER) at a time.
tags: [concept, mental-model, greenlets]
related: [docs/concepts/deferred-execution.md, docs/concepts/interleaver-and-controller.md, docs/concepts/batching-and-invokers.md, docs/developing/interleaver-internals.md]
sources: [src/nnsight/intervention/interleaver.py]
---

# Threading and Mediators

## One thread, taking turns

Each block of intervention code — the body of `with model.trace(x)`, or one
`with tracer.invoke(...)` — is run by a `Mediator` in its own **greenlet** (the
"worker"). Greenlets are not threads: the worker and the model take strict turns on
**one OS thread**.

The worker runs until it needs a value, then *parks*: it switches back to the model
side carrying one event, and waits. The model keeps running; when it reaches the
location the worker asked for, it switches back into the worker with the value. The
worker continues to its next park, or finishes.

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        a = model.transformer.h[0].output.save()   # worker 1 parks until layer 0 runs
    with tracer.invoke("World"):
        b = model.transformer.h[0].output.save()   # worker 2, same forward pass
```

Three consequences:

- **A worker has one pending request at a time**, so it must ask for locations in the
  order the model reaches them. Asking for one the model already passed raises
  [`OutOfOrderError`](../errors/out-of-order-error.md) when the run ends — not a
  deadlock; there is no thread to hang.
- **Workers never run in parallel.** Several invokes share one forward pass and
  resume in the order the model reaches what each asked for, not the order they were
  written. To make one wait for another, use a
  [barrier](../usage/barrier.md).
- **Exceptions keep their type** and surface with a traceback pointing at your line
  ([errors/debug-mode.md](../errors/debug-mode.md)).

## The events

A worker parks by sending one of four events:

| Event | Means |
|-------|-------|
| `VALUE` | "Give me the value at this location." Served when the model reaches it. |
| `SWAP` | "Replace the value at this location with mine." Substituted into the forward. |
| `SKIP` | "Don't run the computation at this location; use my value as its result." Checked before the module runs. |
| `BARRIER` | "Wait for the other blocks." Names no location; released by the last worker to arrive. |

There is no end or exception event: a worker finishing is its greenlet returning,
and an error propagates out of the switch. The location carries an occurrence tag
(`.i0`, `.i1`, …) — see
[interleaver-and-controller.md](interleaver-and-controller.md#occurrences). The
lifecycle and teardown are in
[developing/interleaver-internals.md](../developing/interleaver-internals.md).

## Related

- [Deferred Execution](deferred-execution.md) — how the block becomes something a worker can run.
- [Interleaver and Controller](interleaver-and-controller.md) — the model side.
- [Batching and Invokers](batching-and-invokers.md) — several workers on one batched forward.

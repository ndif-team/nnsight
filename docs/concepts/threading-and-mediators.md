---
title: Threading and Mediators
one_liner: Each block is one Mediator running in a greenlet (not a thread) that parks on a location and switches back to the model side, exchanging typed events (VALUE, SWAP, SKIP, BARRIER) one at a time.
tags: [concept, mental-model, greenlets]
related: [docs/concepts/deferred-execution.md, docs/concepts/interleaver-and-controller.md, docs/concepts/batching-and-invokers.md]
sources: [src/nnsight/intervention/interleaver.py]
---

# Threading and Mediators


## What this is for

A `Mediator` (`interleaver.py`) is the runtime object behind one block of intervention code — the body of a `with model.trace(...)` (direct input) or one `with tracer.invoke(...)`. It owns:

- The compiled block (`code`) and the `Scope` it runs against (`lcls`).
- A **greenlet** (the "worker") that runs the block.
- `pending`: the single event the worker is currently parked on.
- Per-run state: `iteration` / `iterations` (occurrence tracking), `batch_group`, `caches`.

A worker and the model take strict turns on **one OS thread**. The worker runs until it needs a value, then *parks* — a greenlet switch back to the parent (the model side) — carrying an event tuple. There is at most one pending event per mediator at a time. This is why a worker must request locations in the order the model reaches them.

## When to use / when not to use

Structural — every trace uses at least one Mediator. You don't construct one; the tracer does. Read this to understand:

- Why two invokes can't truly run in parallel (they share one forward pass, on one thread).
- Why accessing modules out of forward-pass order raises `OutOfOrderError`.
- What `barrier()`, `stop()`, `skip()`, and `tracer.iter` do underneath.

## Canonical pattern

```python
with model.trace() as tracer:
    # Each invoke -> one Mediator -> one greenlet worker.
    with tracer.invoke("Hello"):
        a = model.transformer.h[0].output.save()   # parks until layer 0 fires

    with tracer.invoke("World"):
        b = model.transformer.h[0].output.save()   # its own worker, same forward
```

## The event protocol

A worker parks by switching a tuple `(Event, location, ...)` to its parent. `Event` (`interleaver.py`) has exactly four members:

| Event | Raised from | Means |
|-------|-------------|-------|
| `VALUE` | `Mediator.value` | "Read the value at this location." Worker parks; the model side serves it via `Mediator.handle` once the location is reached. |
| `SWAP` | `Mediator.swap` | "Replace the value at this location with mine." The model side substitutes it into the forward and resumes the worker. |
| `SKIP` | `Mediator.skip` | "Skip the computation gated at this location, using my value as its result." Queried by a module/op forward wrapper *before* it runs. |
| `BARRIER` | `Mediator.barrier` | "Wait for the other blocks." Names no location, so the model side never serves it — another worker does, on its way past the same barrier. |

There are no `END` or `EXCEPTION` events. A worker finishing is just its greenlet running to completion (falsy afterwards; see `alive`, `interleaver.py`). An exception simply propagates out of the `switch` — with a clean intervention-only traceback stashed on it as `__intervention_tb__` before the model's own frames pile on (`interleaver.py`).

The location a worker parks on is tagged with the occurrence it wants: `"{location}.i{n}"` (`Mediator.event`, `interleaver.py`). With no `tracer.iter`, `n` is always `0`, so every request binds to the first visit — see [occurrence tagging](interleaver-and-controller.md).

## Lifecycle

1. `Interleaver.__enter__` calls `Mediator.start(interleaver)` (`interleaver.py`): it creates the greenlet, stashes a weakref back to the mediator on it (so intervention code can find its own mediator via `getcurrent().mediator()`), and switches in — running the block up to its first park. Whatever it parks on becomes `pending`.
2. The worker hit an `.output` access; the property called `Mediator.value(location)`, which switched `(Event.VALUE, "…output.i0")` to the parent and blocked.
3. The model runs on the main greenlet. When it reaches that module, the controller's `Interleaver.handle` calls `Mediator.handle(provider, value)` (`interleaver.py`), which serves the value and `switch`es back into the worker.
4. `Mediator.handle` loops while the worker keeps parking on the *same* location (e.g. read then swap the same output), then records that the model passed this occurrence (`iterations[provider] += 1`) and returns the possibly-edited value.
5. The worker runs to its next park, or finishes. No teardown event — a finished greenlet is just `alive == False`.
6. `check_dangling_mediators` (`interleaver.py`), called after the model returns, throws `OutOfOrderError` into any worker still parked (or a `ValueError` for an unmet barrier).

## Out-of-order = error

Within one worker, requests happen *in execution order*, and the model runs in forward order. If you read a **later** module's output and then an **earlier** one's, the earlier one has already run past — its next visit will never come. This is detected and raised. Verified:

```python
with model.trace("Hello"):
    out = model.transformer.h[0].output.save()   # output comes late in the block
    args, kwargs = model.transformer.h[0].inputs  # but input already ran
```

```
OutOfOrderError: 'model.transformer.h.0.input.i0' was requested but the model
already ran past it
```

Read a module's `.input` **before** its `.output`. To access modules in a different order, use a separate invoke (a separate worker over the same forward).

## Cross-worker communication

- **Cross-invoke variables**: blocks written in the same frame share their locals through the `Scope`'s `shared` dict (`tracing/util.py`), so a name bound in one `invoke` is visible in a later one — but only after that block has actually run. Because workers resume in *model-reached* order, not definition order, use a **barrier** when one block must read before another writes.
- **Barriers**: `tracer.barrier(n)` returns a `Barrier` (`barrier.py`); each block calls it, parks on `Event.BARRIER`, and the last to arrive releases the rest by `switch`ing each parked worker directly. See [Batching and Invokers](batching-and-invokers.md).
- **`result`**: the model's return value isn't produced by any module. `Envoy.interleave` calls `Interleaver.handle("result", result)` after the forward, serving anything parked on `tracer.result`.

## Early stop

`tracer.stop()` raises `EarlyStopException` inside the worker (`intervention/tracer.py`). It propagates out through the model's forward, unwinding it; `Interleaver.__exit__` swallows it (`interleaver.py`) since the halt was intentional.

## Gotchas

- **Access modules in forward-pass order within one invoke.** Reverse order raises `OutOfOrderError`, not a deadlock — there are no threads to hang.
- **Workers do not run in parallel.** They interleave cooperatively on one thread; two invokes share one forward pass and resume in the order the model reaches what each asked for.
- **A worker finding its own mediator** uses `getcurrent().mediator()` (a weakref set in `start`). `tracer.iter` and `tracer.barrier()` rely on this to reach the running mediator.
- **Exceptions keep their type.** A raised error propagates with its class intact and a filtered, intervention-only traceback (`InterleavingTracer.traceback`).
- **Deferring exceptions** (`Interleaver.defer_exceptions`) is a driver-specific mode (vLLM, whose engine schedules the next step itself): a worker's error is recorded on its mediator rather than raised out of the handoff. Local traces don't use it.

## Related

- [Deferred Execution](deferred-execution.md) — how the worker's block is captured and compiled before it runs.
- [Interleaver and Controller](interleaver-and-controller.md) — the model side: controllers, `Interleaver.handle`, occurrence tagging.
- [Batching and Invokers](batching-and-invokers.md) — multiple workers on one batched forward.
- Source: `src/nnsight/intervention/interleaver.py` (`Event`, `Mediator`, `Interleaver`), `src/nnsight/intervention/barrier.py` (`Barrier`).

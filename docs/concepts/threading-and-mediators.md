---
title: Threading and Mediators
one_liner: Each invoke is a worker thread driven by a Mediator that exchanges typed events (VALUE, SWAP, SKIP, BARRIER, END, EXCEPTION) with the main thread one at a time.
tags: [concept, mental-model, threading]
related: [docs/concepts/deferred-execution.md, docs/concepts/interleaver-and-hooks.md, docs/concepts/batching-and-invokers.md]
sources: [src/nnsight/intervention/interleaver.py:718, src/nnsight/intervention/interleaver.py:949, src/nnsight/intervention/interleaver.py:1207, src/nnsight/intervention/interleaver.py:338]
---

# Threading and Mediators

## What this is for

A `Mediator` (`interleaver.py:718`) is the runtime object behind a single `tracer.invoke(...)`. It owns:

- The compiled intervention function.
- A `Thread` that runs that function (the worker).
- A pair of one-slot queues for synchronous handoff with the main thread.
- Per-mediator state: history, iteration counters, batch group, hooks.

Mediators run **serially** within an `Interleaver`. The model executes on the main thread; a worker thread runs intervention code; values pass between them one at a time via an event protocol.

## When to use / when not to use

This is structural — every trace uses Mediators, including single-invoke traces. You generally don't construct a `Mediator` yourself; the tracer does. Read this doc to understand:

- Why two invokes can't run truly in parallel (they share the model's forward pass).
- Why deadlocks happen if you access modules out of forward-pass order.
- What `barrier()`, `stop()`, `skip()`, `next()` actually do under the hood.

## Canonical pattern

```python
with model.trace() as tracer:
    # Each invoke -> one Mediator -> one worker thread.
    # They run in definition order, serially.
    with tracer.invoke("Hello"):
        a = model.transformer.h[0].output[0].save()  # worker blocks until layer 0 fires

    with tracer.invoke("World"):
        b = model.transformer.h[0].output[0].save()  # this whole invoke runs after the first finishes
```

## The event protocol

The worker thread communicates with the main thread by putting one event on `mediator.event_queue`, then blocking on `mediator.response_queue` until the main thread fulfills it. The events are defined in `Events` (`interleaver.py:338`):

| Event | Sent from | Means |
|-------|-----------|-------|
| `VALUE` | worker | "I want the value at this provider path." Worker blocks; main thread delivers via `Mediator.handle` once the matching hook fires. |
| `SWAP` | worker | "Replace the value at this provider path with this one." Main thread routes through `batcher.swap` and unblocks the worker. |
| `SKIP` | worker | "When module X's pre-forward fires, inject `__nnsight_skip__` so its real forward is bypassed." Used by `Envoy.skip(...)`. |
| `BARRIER` | worker | "Synchronize me with these other mediators here." Once all participants have hit the barrier, all are released. |
| `END` | worker | "I'm finished — drain me." Sent from the try/catch wrapper that `Invoker.compile` adds. |
| `EXCEPTION` | worker | "I crashed; here's the exception." Main thread re-raises (with traceback rewritten to point at user code via `wrap_exception`). |

The two queues are `Mediator.Value` (`interleaver.py:767`), each a single-slot lock-based handoff. Only ever one event in flight per mediator — this is the source of the "must access in forward-pass order" rule.

## Lifecycle

1. `Mediator.start(interleaver)` (`interleaver.py:871`) launches the worker thread, captures the calling thread's CUDA stream so worker-side ops use the same stream, then waits for the first event.
2. The worker hits an `Envoy.output` access; `eproperty.__get__` calls `interleaver.current.request(...)` which sends a VALUE event and blocks.
3. The main thread is running the model. A one-shot PyTorch hook for that module fires, calls `mediator.handle(provider, value)` (`interleaver.py:949`), which iterates pending events and delivers the value into `response_queue`. The worker unblocks.
4. The worker keeps running until it hits the next access (another VALUE, or SWAP for an assignment) or finishes. Each event is paired one-to-one with a hook callback.
5. On normal completion, the try/catch wrapper sends END. `Mediator.handle_end_event` calls `Mediator.cancel`, joining the worker and clearing state.
6. If anything raises, the wrapper sends EXCEPTION. The main thread re-raises in the user's calling thread, with the traceback rewritten to look like it came from the original source.

## Out-of-order = deadlock

Inside one invoke (one worker), the worker requests provider paths *in the order it executes*. The main thread runs the model forward in execution order. If the worker requests layer 5 first and then layer 2, the main thread reaches layer 2's hook first — but the worker is asking for layer 5, so `handle_value_event` sees `provider != requester`, restores the event, and returns `False` (`interleaver.py:1054`). When layer 5 finally fires the worker unblocks, but by then layer 2 has already passed: the next access to layer 2 will never be fulfilled. The next forward pass (or end-of-trace) reports `OutOfOrderError` via `check_dangling_mediators`.

To access modules out of forward-pass order, use a separate invoke (separate worker, separate forward pass through the same batch).

## Cross-mediator communication

- **Cross-invoke variables**: Worker threads `push()` their locals to a shared frame after every event so a later mediator can see them. Controlled by `CONFIG.APP.CROSS_INVOKER` (`interleaver.py:1304`).
- **Barriers**: `tracer.barrier(n)` returns a callable; calling it sends a BARRIER event with the participating mediator names. The mediator that completes the barrier is the one that sees all `n` participants — it then releases all of them via `handle_barrier_event` (`interleaver.py:1123`).
- **Provided values**: Some values aren't produced by a PyTorch hook (e.g. vLLM logits, generation results). `Interleaver.handle(...)` (`interleaver.py:601`) broadcasts these to every mediator and bumps the iteration tracker for that path.

## Hooks lifecycle on a Mediator

Every PyTorch hook a mediator registers (one-shot intervention, persistent cache, persistent iter-tracker, gradient) is appended to `mediator.hooks`. `Mediator.remove_hooks` (`interleaver.py:1354`) drains the list at cancel. `.remove()` is idempotent across all handle types used. See [Interleaver and Hooks](interleaver-and-hooks.md) for the registration side.

## Gotchas

- **Within one invoke, access modules in forward-pass order.** Reverse order deadlocks.
- **Workers do not run in parallel.** Two invokes share the same forward pass; they execute serially in definition order.
- **CUDA stream propagation matters.** `Mediator.start` reads `torch.cuda.current_stream()` on the main thread and calls `torch.cuda.set_stream` in the worker. If you swap streams *after* `start`, the worker won't follow.
- **Iteration counter resolution is dual-mode.** Inside `tracer.iter[i]`, `mediator.iteration` is the explicit target. Outside, the per-path `iteration_tracker` is used. See `Interleaver.iterate_requester` (`interleaver.py:446`).
- **EXCEPTION events use `wrap_exception`** to rewrite the traceback. Catching exceptions by type still works; the exception class is preserved.

## Related

- [Deferred Execution](deferred-execution.md) — how the worker function is built before threading begins.
- [Interleaver and Hooks](interleaver-and-hooks.md) — the hook side of the protocol.
- [Batching and Invokers](batching-and-invokers.md) — multiple mediators on one batch.
- Source: `src/nnsight/intervention/interleaver.py` (`Mediator`, `Interleaver`, `Events`).

---
title: Barrier
one_liner: Cross-invoke synchronization point for sharing values across invokes that touch the same module.
tags: [usage, batching, synchronization]
related: [docs/usage/invoke-and-batching.md, docs/usage/access-and-modify.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/tracing/tracer.py:551, src/nnsight/intervention/tracing/tracer.py:646, src/nnsight/intervention/interleaver.py:1123]
---

# Barrier

## What this is for

Each `tracer.invoke(...)` runs as a separate worker thread, and threads run **serially**. A variable defined in invoke 1 is normally not yet materialized by the time invoke 2 starts referring to it — it lives in invoke 1's worker frame.

`tracer.barrier(n)` is a sync primitive: when all `n` participating invokes call `barrier()`, the interleaver pauses the first to reach it, runs the others up to their barrier call, and then releases everyone together. At that point, variables produced before each invoke's `barrier()` have been pushed back to the shared frame and are visible to other invokes.

You need this whenever **two invokes both access the same module** and you want to share a value across the boundary.

## When to use / when not to use

- Use when invoke 2 needs a value that invoke 1 produced from the same module. Without a barrier, you get `NameError`.
- Don't use when invokes touch entirely different modules — cross-invoker variable sharing handles that case automatically (controlled by `CONFIG.APP.CROSS_INVOKER`).
- Don't use as a substitute for `tracer.stop()` or `module.skip()`.

## Canonical pattern (activation patching)

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)   # 2 participating invokes

    # Clean run
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :]
        barrier()                  # signal: clean_hs is now available

    # Patched run
    with tracer.invoke("The Colosseum is in"):
        barrier()                  # wait until invoke 1 has materialized clean_hs
        model.transformer.h[5].output[0][:, -1, :] = clean_hs
        patched = model.lm_head.output.save()
```

## Why a barrier is required here

Both invokes touch `transformer.h[5].output`. The mediator threads run serially, and the second invoke's mediator does **not** automatically wait for the first invoke's mediator to finish — it only waits when it requests its own value. The simple "cross-invoker push" mechanism (which works when invokes touch different modules) is not sufficient because the first invoke is still mid-flight when the second tries to access the shared module.

The barrier introduces an explicit synchronization point: invoke 1 pauses, the interleaver advances invoke 2 to its own `barrier()` call, then both proceed. By that time, invoke 1's locals have been pushed (`Mediator.push`, `interleaver.py:1304`) and `clean_hs` exists in the shared frame.

## How it works

`InterleavingTracer.barrier` (`src/nnsight/intervention/tracing/tracer.py:551`) returns a `Barrier` object (`tracer.py:646`):

```python
class Barrier:
    def __init__(self, model, n_participants):
        ...
    def __call__(self):
        mediator = self.model.interleaver.current
        self.participants.add(mediator.name)
        if len(self.participants) == self.n_participants:
            participants = self.participants
            self.participants = set()
            mediator.send(Events.BARRIER, participants)
        else:
            mediator.send(Events.BARRIER, None)
```

`Mediator.handle_barrier_event` (`interleaver.py:1123`) is called by the last participant:

```python
def handle_barrier_event(self, provider, participants):
    if participants is not None:
        for mediator in self.interleaver.mediators:
            if mediator.name in participants:
                self.interleaver.current = mediator
                mediator.respond()
                mediator.handle(provider, ...)
    return False
```

Each participating mediator is woken up in order, its `respond()` releases its waiting worker, and `handle()` lets it continue past the barrier.

## Multiple barriers

A single `Barrier` instance is **reusable** — its `participants` set is reset to empty after firing. So you can use the same barrier multiple times in a single trace:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)

    with tracer.invoke("A"):
        h_a_5 = model.transformer.h[5].output[0]
        barrier()
        h_a_8 = model.transformer.h[8].output[0]
        barrier()

    with tracer.invoke("B"):
        barrier()
        x = h_a_5  # use after first barrier
        barrier()
        y = h_a_8  # use after second barrier
```

If you need different `n_participants` at different points, create separate barriers.

## Gotchas

- `n_participants` must equal the actual number of invokes that will call `barrier()`. If fewer call it, the barrier never fires and the trace deadlocks.
- The barrier returned by `tracer.barrier(n)` is a value, not a context manager — call it as `barrier()`.
- Cross-invoker sharing without a barrier works only when the shared variable is defined in invoke 1 from a module **not** also accessed in invoke 2. Otherwise: `NameError`.
- A barrier is per-trace — defining one outside the trace context is meaningless.
- Forgetting to call the barrier in one invoke (e.g. early `return`) hangs the trace.

## Related

- `docs/usage/invoke-and-batching.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/trace.md`

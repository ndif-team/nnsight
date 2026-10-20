---
title: Interleaver Internals
one_liner: How Interleaver and Mediator coordinate the model's forward pass with greenlet-worker interventions via park/switch and the VALUE/SWAP/SKIP/BARRIER protocol.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/controller.md, docs/developing/batching-internals.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/iterator.py, src/nnsight/intervention/barrier.py]
---

# Interleaver Internals

## What this covers

The runtime that pairs a PyTorch forward pass with one greenlet worker per trace
block. Two classes, both in `src/nnsight/intervention/interleaver.py`:

- `Interleaver` — one per `Envoy` tree; owns the list of workers, the per-location
  indexes the controllers consult, and the visit counts; persists for the model's
  lifetime.
- `Mediator` — one per block (per `tracer.invoke(...)`, or the whole trace body,
  or a registered edit); runs the block in a greenlet and drives the park/switch
  interaction with the model side.

## Architecture

### Park / switch

```
model side (parent greenlet)              worker (child greenlet, one per block)
----------------------------              --------------------------------------
model.forward(...)                        exec(block)
   |                                          |
   | controller hands off                     | reads model.layer.output
   v                                          v
interleaver.handle(loc, value)  <---.         Mediator.value("model.layer.output")
   |                                 \        |
   | mediator.handle(loc, value)     |        | worker.parent.switch(
   |   pending matches this visit?   |        |     Pending(VALUE, loc, occurrence))
   |   YES: switch served value in --------->  (worker resumes with the value)
   |   NO:  leave parked, return              |
   v                                          v
(value flows back into forward)           (.save(); run to next park or finish)
```

`Mediator.event` is the worker-side primitive: it switches to the parent greenlet
handing over a `Pending` and blocks until the parent switches a value back.
`Mediator.switch` is the parent-side counterpart: it resumes the worker with a
value and returns whatever the worker parks on next, or `None` when the worker
finishes (a greenlet is falsy once it has run to completion, which is what
`Mediator.alive` tests).

If the worker raises, `Mediator.switch` stashes its traceback on the exception as
`__intervention_tb__` before the model's frames unwind on top; `Tracer.traceback`
and `errors.capture_exception` prefer it, so the user sees their own line. Code
running inside a worker reaches its own mediator with `getcurrent().mediator()`
(a weakref set on the greenlet); `Mediator.current` wraps it, raising `Cannot
access ... outside of interleaving` when no worker is running.

### The event protocol

`Event` has four members:

| Event | Park call | Served by |
|---|---|---|
| `VALUE` | `Mediator.value(loc)` | the model (the controller's handoff) |
| `SWAP` | `Mediator.swap(loc, v)` | the model |
| `SKIP` | `Mediator.skip(loc, v)` | the model (the controller's skip gate) |
| `BARRIER` | `Mediator.barrier()` | another worker |

A finished worker returns `None` from `switch`; a raised exception propagates
through `switch`. `BARRIER` names no location the model produces, so the model side
never serves it — the last block to reach a shared `Barrier` releases the rest.

### Pending and locations

What a worker is parked on is a `Pending(event, provider, iteration, value)`: the
location undecorated (`"model.layer.output"`), the occurrence it wants, and the
replacement a swap or skip carries. It prints as `'model.layer.output.i2'`, the
form worth reading in an error.

A provider is a plain string. The controller emits `"{path}.input"`,
`"{path}.skip"` and `"{path}.output"` for a module; `"result"` is the model's
return value; operation-level ones are `"{path}.source.relu_0.output"` and so on
([source-internals.md](source-internals.md)).

### Occurrences

A location can be reached many times in one run — a module is revisited on every
step of a generation loop. The interleaver keeps **one counter per location**,
`counts[provider]`, bumped once per visit after everyone parked on it has been
served. A worker remembers the counts when it started (`counts_at_start`), so its
own occurrence of a location is a subtraction: `occurrence(loc) = counts[loc] −
counts_at_start[loc]`. That is what `tracer.iter[n]` pins against, and what lets a
worker that joins a run late (a vLLM request scheduled mid-stream, a preempted one
resuming) count from its own start.

When a worker parks, `Mediator.event` records the occurrence it wants:

- `iteration` is an int (pinned) — the step chosen by `tracer.iter[n]`.
- `iteration is None` (relaxed) — the next occurrence the model hasn't handled,
  `occurrence(provider)`, so the request follows the model sequentially.
- With no `tracer.iter`, `iteration` stays `0`, so every request is occurrence 0.

After the first hit of a pinned non-zero step the mediator relaxes to `None`, so
the rest of that step's requests follow the model rather than re-forcing the
index. `Iterations.__iter__` (`iterator.py`) walks the pointer across a range of
steps.

## The Mediator class

`Mediator`'s fields (the block and its `Scope`, `batch_group`, `worker`, `pending`,
`iteration`, `counts_at_start`, `caches`, ...) are listed in the `Attributes:`
section of its docstring in `src/nnsight/intervention/interleaver.py`.

### Mediator.handle — draining a visit

Given a visit of `provider`, `Mediator.handle` returns immediately unless the
worker is parked on this provider at this occurrence. Otherwise it loops: a
`VALUE` is served the value (narrowed to the worker's rows when batching); a
`SWAP` replaces the value with the worker's (widened back into the batch); a
`SKIP` gathers the worker's replacement. Each `switch` returns the worker's next
`Pending`; the loop continues while it is still this visit — a worker may read a
location and then assign it before parking elsewhere or finishing. See
[batching-internals.md](batching-internals.md) for `narrow`/`widen`.

## The Interleaver class

`Interleaver`'s fields — `mediators`, `batcher`, the `interleaving`/`busy` gate the
controller reads, the `watched` and `observers` indexes, `counts`, `fragments`,
`defer_exceptions` — are documented in its docstring in the same file.
`Interleaver.instrument(envoy)` installs a module's controller; see
[controller.md](controller.md).

### reindex

`reindex()` rebuilds `observers`, `watched` and `busy` from `mediators`. `watched`
is every location a visit has to stop for: one some worker is parked on, or one a
cache keeps. `__enter__` calls `reindex` once the workers have parked; a driver
that replaces the list while a run is in progress (a scheduler reshuffling which
workers are in the batch) calls it itself afterwards, since nothing else notices
the swap. `watched` is also rebuilt by `handle` after it has served anyone — the
only other time a worker can park.

### handle — one visit

`Interleaver.handle(provider, value)` is what the controller calls for a location
in `watched` (for one that is not, the controller bumps `counts` itself and never
reaches here; a driver with no controller — an engine value, a gradient hook —
calls it for every visit):

1. **Fast path.** If `provider` is not in `watched`, bump `counts[provider]` and
   return the value untouched.
2. **Whole.** If the runtime's `fragments` say the value is a shard or partial sum
   and a worker will be served on this occurrence (or a cache keeps it), make it
   whole — one collective per visit, on every rank alike.
3. **Serve.** While some worker is parked here at this occurrence (`_ready`), offer
   the value to each; serving one can release another into parking here (a
   barrier), so loop until nobody is. Under `defer_exceptions` a worker's error is
   recorded on it and its `pending` cleared instead of raised.
4. Bump `counts[provider]`. A batched skip's gathered replacements are assembled
   into one value.
5. **Observe.** Feed the post-intervention value to the subscribed caches, narrowed
   to each worker's rows.
6. **Fragment.** Hand back the runtime's piece, carrying whatever the workers left.

Workers parked on the same visit are served in `mediators` order, so if two invokes
both edit one location, invoke 0's edit lands before invoke 1's — definition order.

### __enter__ / __exit__ / cancel

`__enter__` flips `interleaving = True`, clears `sourced`, `start()`s every worker
that has no greenlet yet so each parks on its first requested location, then
`reindex()`es. If a worker errors on start it resets the flag so it doesn't leak to
the next run.

`__exit__` flips `interleaving` back and returns `True` for an `EarlyStopException`
(from `tracer.stop()`) to swallow it — an intentional halt. It does not clear the
mediators; that is `cancel`'s job.

`cancel` releases each worker's greenlet, empties `mediators` and drops the
`batcher`, so the next run starts clean. The controllers stay installed. A worker
still `alive` — parked because the model's forward raised before reaching its
location — is thrown a `GreenletExit` first: dropping the reference does not end a
greenlet, and a parked one holds its frame, the block's scope and through it the
model. An exception out of the block's own `finally` warns rather than raising, so
it cannot hide the error that ended the run.

### check_dangling_mediators

After the model finishes, `Envoy.interleave` calls `check_dangling_mediators()`.
Any worker still `alive` was waiting for something that never came:

- A `BARRIER` never released (fewer blocks reached it than its count) → throw a
  `ValueError` into the worker pointing at the waiting line.
- A plain request (`iteration == 0`) for a location the model ran past or never
  called → throw `OutOfOrderError` into the worker, so the traceback points at the
  waiting line:

  ```text
  'model.transformer.h.2.output.i0' was requested but the model already ran past it
  ```

- A request inside an open-ended `tracer.iter[:]` loop (`iteration != 0`) for a
  step the model never ran → throw `OutOfOrderError` to unwind the worker (running
  its `finally` blocks), catch it, and `warnings.warn` instead of raising. Values
  from steps that *did* run are already saved.

> This is why an open-ended `for step in tracer.iter[:]:` unwinds the loop — **and
> every line after it** — when the model stops. To keep trailing code, bound the
> loop (`tracer.iter[:N]` matching `max_new_tokens`).

## Barriers

`Barrier` (`barrier.py`), from `tracer.barrier(n)`, is a meeting point for `n`
blocks. Each block calls it; a block that isn't the last parks via
`Mediator.barrier()`. The last one in doesn't park — it releases the others by
`switch()`ing each waiting worker directly, then carries on. A worker released this
way parks on the same visit its releaser is being served in, and `handle`'s settle
loop serves it there. Use a barrier when one block hands a value to another
(activation patching across invokes), so the write is guaranteed to happen after
the read.

## `tracer.stop()` and early exit

`InterleavingTracer.stop` raises `EarlyStopException` from inside the worker; it
propagates through `switch` into `Interleaver.handle`, which treats it as control
flow rather than an error: it holds the exception, clears that worker's `pending`,
finishes the visit (count, `assemble_skip`, the observing caches, the other workers
parked here, the fragment `undo`) and re-raises it on the way out, unwinding the
model's forward from a completed visit. So the location a stop fires at is served
and recorded, and only what follows it is lost. `Interleaver.__exit__` swallows it.
If a worker stops before the model even starts (during `__enter__`'s `start`),
`Envoy.interleave` catches it directly.

## Caches

`tracer.cache(...)` registers a `Cache` on the calling worker's mediator, and
`reindex()` records it in `observers` under each location it keeps. `handle`'s
observe step feeds each such location's post-intervention value to its caches,
narrowed to the worker's batch rows. A cache sees exactly what the controllers
surface; nothing extra is installed on the module, and an observed location leaves
the fast path even when no worker is parked there.

## Key files / classes

- `src/nnsight/intervention/envoy.py` — `Envoy.interleave`, which runs the model
  and the workers.
- `src/nnsight/intervention/barrier.py` — `Barrier`;
  `src/nnsight/intervention/iterator.py` — `Iterations.__iter__`.

## Lifecycle / sequence

The interleaver's part of the [trace lifecycle](./architecture-overview.md#lifecycle--sequence)
(its steps 6-8), for `with model.trace("hi"): hidden = model.layer.output.save()`:

1. `with self.interleaver:` → `__enter__` `start()`s the one worker; the block runs
   to `model.layer.output` and parks as `Pending(VALUE, "model.layer.output", 0)`.
2. The model runs. `model.layer`'s controller calls
   `interleaver.handle("model.layer.output", output)`; the worker is parked here
   at occurrence 0, so `switch` hands the value in.
3. The worker `.save()`s and finishes → `switch` returns `None`; `alive` is `False`.
4. `interleave` serves `"result"` (nobody parked), `check_dangling_mediators()` is
   a no-op, and the interleaver exits and `cancel`s.

## Extension points

New served values are in [extending-envoy.md](extending-envoy.md). For a custom
driver:

- **Keep running past a worker's error.** Set `interleaver.defer_exceptions = True`
  and read `mediator.exception` after each step to end that worker's request
  without tearing down the run.
- **Reschedule workers per step.** Assign `interleaver.mediators` and call
  `reindex()`; give each worker a `batch_group` for its rows.
- **Shard values.** Supply a `Fragments` (`intervention/fragments.py`) that says
  which locations are pieces and how to make them whole.
- **A module the runtime calls without `forward`.** Its controller never runs;
  have the driver call `interleaver.handle(location, value)` directly.

## Related

- [controller.md](controller.md) — how a module reaches `handle`.
- [batching-internals.md](batching-internals.md) — `narrow`/`widen`/`gather_skip`.
- [source-internals.md](source-internals.md) — operation-level locations.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md),
  [docs/concepts/interleaver-and-controller.md](../concepts/interleaver-and-controller.md)
  — the mental-model versions.

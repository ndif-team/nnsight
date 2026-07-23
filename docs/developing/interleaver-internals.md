---
title: Interleaver Internals
one_liner: How Interleaver and Mediator coordinate the model's forward pass with greenlet-worker interventions via park/switch and the VALUE/SWAP/SKIP/BARRIER protocol.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/hook-system.md, docs/developing/batching-internals.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/iterator.py, src/nnsight/intervention/barrier.py]
---

# Interleaver Internals

## What this covers

The runtime that pairs a PyTorch forward pass with one greenlet worker per trace
block. Two classes, both in `src/nnsight/intervention/interleaver.py`:

- `Interleaver` (`:430`) — one per `Envoy` tree; owns the forward hooks and the
  list of workers; persists for the model's lifetime.
- `Mediator` (`:93`) — one per block (per `tracer.invoke(...)`, or the whole trace
  body, or a registered edit); runs the block in a greenlet and drives the park/
  switch interaction with the model side.

Because a worker and the model take strict turns **on one thread**, there are no
locks, no queues, and no worker threads — only greenlet switches. This is the
single biggest change from older nnsight, which used one OS thread per invoke plus
single-slot lock queues.

## Architecture

### Park / switch, not threads and queues

```
model side (parent greenlet)              worker (child greenlet, one per block)
----------------------------              --------------------------------------
model.forward(...)                        exec(block)
   |                                          |
   | forward hook fires                       | reads model.layer.output
   v                                          v
interleaver.handle(loc, value)  <---.         Mediator.value("model.layer.output")
   |                                 \        |
   | mediator.handle(loc, value)     |        | worker.parent.switch(
   |   pending == this location?     |        |     (Event.VALUE, "....output.i0"))
   |   YES: switch served value in --------->  (worker resumes with the value)
   |   NO:  leave parked, return              |
   v                                          v
(value flows back into forward)           (.save(); run to next park or finish)
```

`Mediator.event` (`:245`) is the worker-side primitive: it switches to the parent
greenlet handing over an event tuple and blocks until the parent switches a value
back. `Mediator.switch` (`:345`) is the parent-side counterpart: it resumes the
worker with a value and returns whatever the worker parks on next (or `None` when
the worker finishes — a greenlet is falsy once it has run to completion, which is
what `Mediator.alive`, `:313`, tests).

### The Event protocol

`Event` (`:56`) has exactly four members — the OLD `END`/`EXCEPTION` events are
gone (a finished worker just returns `None` from `switch`; a raised exception
propagates through `switch`):

| Event | Park call | Tuple | Served by |
|---|---|---|---|
| `VALUE` | `Mediator.value(loc)` (`:270`) | `(VALUE, loc)` | the model (a forward hook) |
| `SWAP` | `Mediator.swap(loc, v)` (`:280`) | `(SWAP, loc, v)` | the model |
| `SKIP` | `Mediator.skip(loc, v)` (`:291`) | `(SKIP, loc, v)` | the model (a skip gate) |
| `BARRIER` | `Mediator.barrier()` (`:302`) | `(BARRIER, None)` | another worker |

`BARRIER` is the odd one: it names no location the model produces, so the model
side never serves it — the last block to reach a shared `Barrier` releases the rest
(see below).

### Locations

A location is a plain provider string. The two the forward hooks emit are
`"{path}.input"` and `"{path}.output"`; others are `"result"` (the model's return
value), `"{path}.skip"` (the skip gate), and operation-level ones like
`"{path}.source.relu_0.output"` (see `source-internals.md`). Envoy
properties are thin wrappers: `envoy.output` is `Mediator.value("{path}.output")`,
`envoy.output = x` is `Mediator.swap("{path}.output", x)`.

### Iteration tagging

A location can be reached many times in one run — e.g. a module revisited on every
step of a generation loop. Each park is tagged with the occurrence the worker wants:
`Mediator.event` appends `.i{n}` to the location (`:267`), where `n` comes from the
worker's `iteration` pointer:

- `iteration` is an int (pinned) — the step chosen by `tracer.iter[n]`.
- `iteration is None` (relaxed) — resolves to `iterations[location]`, the
  mediator's current count for that location (the next occurrence the model hasn't
  handled), so the request follows the model sequentially.
- With no `tracer.iter`, `iteration` stays `0`, so every request is `.i0` — the
  original single-pass behavior.

`Mediator.handle` (`:375`) is where a visit is matched: this visit is the
`iterations[provider]`-th, so it tags itself `{provider}.i{n}` and serves any
pending event whose location string equals that tag — a single string match. A
request pinned to a later step simply doesn't match yet and waits while earlier
visits pass by. After the first hit of a pinned non-zero step, the mediator relaxes
to `None` so the rest of that step's requests follow the model sequentially rather
than re-forcing the index. `Iterations.__iter__` (`iterator.py:111`) is what walks
the `iteration` pointer across a range of steps.

## The Mediator class

Key attributes (`:150`):

```python
code            # the captured block, compiled
glbls, lcls     # the block's globals and its Scope (capture-time names + shared frame)
copy            # exec against a fresh copy of lcls each run (edits only)
node            # AST node, kept so an edit can serialize; None server-side
interleaver     # the run this worker belongs to (set in start)
batch_group     # [start, size] row range in the combined batch, or None (whole batch)
worker          # the greenlet, or None before start / falsy after finish
pending         # the event the worker is parked on, or None
iteration       # which occurrence the worker wants (int pinned, None relaxed)
iterations      # per-location count of visits so far this run
caches          # tracer.cache() caches registered on this worker
exception       # set only under a deferring interleaver (vLLM); see below
```

### Mediator.handle — draining a visit

```python
def handle(self, provider, value):
    location = f"{provider}.i{self.iterations[provider]}"
    batcher = None if self.interleaver is None else self.interleaver.batcher
    while self.pending is not None and self.pending[1] == location:
        if self.iteration:          # first hit of a pinned n>0 step: relax
            self.iteration = None
        if self.pending[0] is Event.VALUE:
            served = value if batcher is None else batcher.narrow(value, self.batch_group)
            self.pending = self.switch(served)
        elif self.pending[0] is Event.SWAP:
            value = self.pending[2] if batcher is None else batcher.widen(value, self.batch_group, self.pending[2])
            self.pending = self.switch()
        elif self.pending[0] is Event.SKIP:
            value = self.pending[2] if batcher is None else batcher.gather_skip(value, self.batch_group, self.pending[2])
            self.pending = self.switch()
    self.iterations[provider] += 1
    return value
```

(`:375`.) The `while` loop lets one worker do several things at the same location
in turn — read it, then assign it — before parking somewhere else or finishing.
Batching, when active, scopes the served value to the worker's rows (`narrow`) and
splices its edit back into the full batch (`widen`); see
`docs/developing/batching-internals.md`. After the visit, `iterations[provider]` is
bumped so the next visit is tagged as the following occurrence.

## The Interleaver class

Attributes (`:468`):

```python
handles          # module path -> [pre_hook_handle, forward_hook_handle]
mediators        # the workers to serve this run
batcher          # the run's Batcher, or None; owns row scoping
interleaving     # True between __enter__ and __exit__; hooks pass through when False
sourced          # op-location -> instrumented callable (per-run; source.py)
defer_exceptions # vLLM: record a worker's error instead of raising out of the hook
```

### instrument — installing the forward hooks

`Interleaver.instrument(envoy)` (`:521`) is called from `Envoy.__init__` (and again
from `Envoy._update` when weights are swapped in). It:

1. Calls `install_skip(envoy)` (`source.py:437`) to register the source/skip
   controller on the module and record this interleaver on it.
2. Removes any existing hooks for the module's path (`remove`, `:663`).
3. Registers a `forward_pre_hook` and a `forward_hook` (both `with_kwargs=True`):

```python
def pre_forward(module, args, kwargs):
    if not self.interleaving:
        return None
    return self.handle(f"{path}.input", (args, kwargs))   # returning (args,kwargs) edits input

def forward(module, args, kwargs, output):
    if not self.interleaving:
        return None
    return self.handle(f"{path}.output", output)          # returning a value edits output
```

The critical property, and the reason the OLD "lazy one-shot hook" doc is wrong for
this codebase: **the hooks are installed once, at instrument time, and stay
installed.** They pass everything through untouched when `interleaving` is `False`,
so an instrumented model runs normally outside a trace. See
`docs/developing/hook-system.md`.

### handle — the fan-out

`Interleaver.handle(provider, value)` (`:566`) offers `value` at `provider` to
every mediator in order and returns it, edited if any wrote to that location:

```python
for mediator in self.mediators:
    try:
        value = mediator.handle(provider, value)
    except Exception as exception:
        mediator.exception = exception
        if not self.defer_exceptions:
            raise
# a batched skip gathered per-invoke replacements; assemble them
if self.batcher is not None:
    value = self.batcher.assemble_skip(value)
# feed the post-intervention value to any tracer.cache() on each worker
for mediator in self.mediators:
    for cache in mediator.caches:
        cache.observe(provider, self.batcher.narrow(value, mediator.batch_group) if self.batcher else value)
return value
```

The workers are served in `mediators` order, so if two invokes both edit the same
location, invoke 0's edit lands before invoke 1's — matching definition order. This
ordering is the list order, not a per-hook priority; there is no
`add_ordered_hook`/`mediator_idx` machinery in this codebase.

Caches observe values **after** interventions have edited them, and see only their
own worker's rows. `defer_exceptions` supports drivers (vLLM) whose forward is one
step of a run they don't control: a worker's error is recorded on its mediator
instead of tearing down the shared engine.

### __enter__ / __exit__

`__enter__` (`:489`) flips `interleaving = True`, clears `sourced`, and `start()`s
every not-already-alive worker so each parks on its first requested location. If a
worker errors on start (e.g. calling `invoke` mid-run), it resets `interleaving` so
the flag doesn't leak to the next run.

`__exit__` (`:509`) flips `interleaving` back to `False` and returns `True` for an
`EarlyStopException` (from `tracer.stop()`) to swallow it — an intentional halt, not
an error. It does **not** clear the mediators; that is `cancel`'s job.

### check_dangling_mediators

After the model finishes, `Envoy.interleave` calls
`check_dangling_mediators()` (`:605`). Any worker still `alive` was waiting for
something that never came:

- A `BARRIER` never released (fewer blocks reached it than its count) → throw a
  `ValueError` into the worker pointing at the waiting line.
- A plain request (`iteration == 0`) for a location the model ran past or never
  called → throw `OutOfOrderError` (`:83`) into the worker, so the traceback points
  at the exact waiting line:

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

### cancel / remove / clear

`cancel` (`:654`) clears `mediators` and `batcher` so the next run starts clean.
`remove(path)`/`clear` (`:663`/`:668`) remove the forward hooks; `__del__` calls
`clear` so hooks drop when the interleaver is GC'd. There is no per-mediator hook
list to drain — the hooks live on the interleaver (installed once), not on
mediators, so worker teardown is just dropping the greenlet.

## Barriers

`Barrier` (`barrier.py:35`), from `tracer.barrier(n)`, is a meeting point for `n`
blocks. Each block calls it; a block that isn't the last parks via
`Mediator.barrier()`. The last one in doesn't park — it releases the others by
`switch()`ing each waiting worker directly (`barrier.py:63`), then carries on. Use a
barrier when one block hands a value to another (activation patching across
invokes), so the write is guaranteed to happen after the read.

## `tracer.stop()` and early exit

`InterleavingTracer.stop` (`tracer.py:89`) raises `EarlyStopException` (`:74`) from
inside the worker; it propagates through `switch` and unwinds the model's forward.
`Interleaver.__exit__` swallows it. If a worker stops before the model even starts
(during `__enter__`'s `start`), `Envoy.interleave` catches it directly (`envoy.py:654`).

## Key files / classes

- `src/nnsight/intervention/interleaver.py:56` — `Event`. VALUE / SWAP / SKIP / BARRIER.
- `:93` — `Mediator`. One greenlet worker.
- `:245` — `Mediator.event`. Worker-side park (switch to parent, tag with iteration).
- `:270`/`:280`/`:291`/`:302` — `value`/`swap`/`skip`/`barrier`.
- `:323` — `Mediator.start`. Create the greenlet, run to first park.
- `:345` — `Mediator.switch`. Parent-side resume; stashes intervention traceback.
- `:375` — `Mediator.handle`. Drain a visit's events.
- `:430` — `Interleaver`.
- `:489`/`:509` — `Interleaver.__enter__`/`__exit__`.
- `:521` — `Interleaver.instrument`. Install the pre/forward hooks + skip controller.
- `:566` — `Interleaver.handle`. Fan-out to every mediator, then caches.
- `:605` — `check_dangling_mediators`. Out-of-order / iter-overrun surfacing.
- `src/nnsight/intervention/envoy.py:612` — `Envoy.interleave`. Runs the model + workers.
- `src/nnsight/intervention/barrier.py:35` — `Barrier`.
- `src/nnsight/intervention/iterator.py:111` — `Iterations.__iter__`. Walks `iteration`.

## Lifecycle / sequence

For `with model.trace("hi"): hidden = model.layer.output.save()`:

1. `Envoy.interleave` prepends any edits, then `with self.interleaver:` →
   `__enter__` sets `interleaving=True` and `start()`s the one worker.
2. `start` switches into the greenlet; the block runs to
   `model.layer.output` → `Mediator.value("model.layer.output")` →
   `event` tags it `model.layer.output.i0` and switches back. `pending` is now
   `(VALUE, "model.layer.output.i0")`.
3. The model runs. `model.layer`'s forward hook fires
   `interleaver.handle("model.layer.output", output)` → the worker's
   `handle` matches `.i0`, `switch`es the value in.
4. The worker resumes, `.save()`s, and runs to the end of the block → `switch`
   returns `None`; `pending` is cleared; `alive` is now `False`.
5. The forward finishes; `interleave` serves the return at `"result"` (no one is
   parked on it), `check_dangling_mediators()` is a no-op (worker done), and the
   interleaver exits and `cancel`s.

## Extension points

- **Serving a value the model doesn't produce.** Call `interleaver.handle(location,
  value)` from your driver after computing it, and expose it as an `Envoy` property
  that reads `Mediator.value(location)`. This is how `tracer.result` and vLLM's
  `.logits`/`.samples` work — no new `Event` type needed.
- **A driver that keeps running past a worker's error.** Set
  `interleaver.defer_exceptions = True` and read `mediator.exception` after each
  step to end that worker's request without tearing down the run (vLLM's pattern).
- **Custom iteration semantics.** The occurrence tag is computed in
  `Mediator.event` from `iteration`/`iterations`; a subclass could resolve
  occurrences differently for a streaming runtime.

## Related

- `docs/developing/hook-system.md` — how the forward hooks are installed and idle.
- `docs/developing/batching-internals.md` — `narrow`/`widen`/`gather_skip`.
- `docs/developing/source-internals.md` — operation-level locations.
- `docs/concepts/threading-and-mediators.md`, `docs/concepts/interleaver-and-hooks.md`
  — the mental-model versions.

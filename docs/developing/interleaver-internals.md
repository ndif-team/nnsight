---
title: Interleaver Internals
one_liner: How Interleaver and Mediator coordinate the model's forward pass with worker-thread interventions.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/lazy-hook-system.md, docs/developing/eproperty-deep-dive.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/hooks.py, src/nnsight/intervention/tracing/iterator.py]
---

# Interleaver Internals

## What this covers

The runtime that pairs a PyTorch forward pass with one worker thread per `tracer.invoke(...)`. Two classes:

- `Interleaver` (`src/nnsight/intervention/interleaver.py:375`) — singleton per `Envoy`; persists for the lifetime of the model wrapper.
- `Mediator` (`src/nnsight/intervention/interleaver.py:718`) — one per invoke; owns a worker thread, an event queue, a response queue, an iteration tracker, and a hook list.

The interleaver does not itself install any hooks (other than the sentinel forward hook in `wrap_module`). Hooks are installed by mediators on demand via the `requires_*` decorators on `eproperty` stubs. Once installed, those hooks call back into `mediator.handle(provider, value)`, which is the central dispatcher this doc walks through.

## Architecture

### Two threads, two queues

```
main thread                                worker thread (one per invoke)
-----------                                -----------------------------
model.forward(...)                         intervention(mediator, info)
   |                                          |
   | hook fires                               | reads model.layer.output
   v                                          v
mediator.handle(prov, val)  <-----.           mediator.request("...output.i0")
   |                              \           |
   | matches event_queue?         |           | event_queue.put((VALUE, "...output.i0"))
   |                              |           | response_queue.wait()
   | YES: respond(value)          |           |
   |                              |           |
   '--> response_queue.put(value) ------------> response_queue.get()
                                              |
                                              v
                                          (worker resumes with value)
```

`Mediator.Value` (`src/nnsight/intervention/interleaver.py:767`) is a single-slot queue built on `_thread.allocate_lock()`. Both the event queue and the response queue use it; only ever one item can be pending in either direction. This is what gives nnsight its serial, deadlock-prone-but-debuggable execution model: at any moment, exactly one thread is active, and the other is waiting on a lock.

### The Interleaver class

```python
class Interleaver:
    mediators: List[Mediator]      # one per invoke
    tracer: InterleavingTracer
    batcher: Batcher
    current: Mediator              # the mediator main-thread is currently servicing
    _interleaving: bool
```

`Interleaver.__enter__` (`interleaver.py:567`) flips `_interleaving = True` and calls `mediator.start(self)` on every mediator. `start()` launches the worker thread and immediately calls `mediator.handle()` to drain the worker's first event — this is how a mediator that finishes immediately (e.g. an empty invoke that just calls `tracer.stop()`) is cleared before the model even runs.

`Interleaver.__exit__` (`interleaver.py:589`) flips the flag back, prunes dead mediators, and swallows `EarlyStopException`. It does *not* call `cancel()` — that is `Envoy.interleave`'s job, in a `finally` block, after `check_dangling_mediators` has been called.

`Interleaver.cancel` (`interleaver.py:424`) iterates every mediator, calls `mediator.cancel()` (which kills the worker by raising `Cancelation`) and `mediator.remove_hooks()` (which drains every hook handle from `mediator.hooks` — see `docs/developing/lazy-hook-system.md`).

### Interleaver.handle — the fan-out

`Interleaver.handle(provider, value, iterate=False)` (`interleaver.py:601`) is used for **provider-side pushes** — values that don't come from a PyTorch forward hook but are produced from the runtime side. Examples:

- `Envoy.interleave` calling `self.interleaver.handle("result", result)` after the model returns, which fulfills `InterleavingTracer.result`.
- `eproperty.provide(obj, value)` for runtime-side values like vLLM's `logits` and `samples`.
- The `operation_fn_hook` swap path that delivers a replaced fn back to the source-tracing pipeline.

It iterates every mediator and calls `mediator.handle(provider, value)` on each. The optional `iterate=True` flag appends the per-mediator iteration counter (`mediator.iteration_tracker[provider]`) to the provider string and bumps it after each handle — this is how multi-step generation values (e.g. `samples.i0`, `samples.i1`) flow without a PyTorch hook.

The return value is the output of the *last* mediator's `handle()`. This matters for `operation_fn_hook`: the worker's `swap()` event delivers the injected function back through `mediator.handle`, which puts it in `batcher.current_value`, which `Interleaver.handle` returns to `wrap_operation` to use as the actual `fn`.

### iterate_requester — single source of truth for iteration

When a worker accesses `model.layer.output`, `eproperty.__get__` builds a base requester `model.layer.output` and then asks the interleaver to suffix it with the current iteration:

```python
def iterate_requester(self, requester: str):
    mediator = self.current
    iteration = (
        mediator.iteration
        if mediator.iteration is not None
        else mediator.iteration_tracker[requester]
    )
    return f"{requester}.i{iteration}"
```

Two sources:

- `mediator.iteration` is set to a concrete int by `IteratorTracer.__iter__` while inside `for step in tracer.iter[N]:`. Use case: "I want layer X's output specifically at generation step N."
- `mediator.iteration_tracker[requester]` is a `defaultdict(int)` bumped by the persistent iter hooks registered by `IteratorTracer.register_iter_hooks` (and by `Interleaver.handle(..., iterate=True)`). It tracks "how many times has this provider already fired in this trace?"

The dual-source resolution lets the same `model.layer.output` syntax work both inside and outside an iter loop. Inside, the explicit `iteration` wins; outside, the tracker's "next free index" is the target.

After a non-zero `iteration` matches, the one-shot input/output hooks set `mediator.iteration = None` (see `hooks.py:206`-`207`) so subsequent accesses fall back to tracker-mode. This is how `tracer.iter[N]` doesn't permanently lock the mediator into step N.

### wrap_module — the only thing the interleaver installs eagerly

`Interleaver.wrap_module(module)` (`interleaver.py:481`) is called by `Envoy.__init__` for every wrapped module. It does two things, exactly once per module:

1. **Replace `module.forward` with a thin wrapper** (`nnsight_forward`):
   ```python
   def nnsight_forward(*args, **kwargs):
       if "__nnsight_skip__" in kwargs:
           return kwargs.pop("__nnsight_skip__")
       source_accessor = getattr(m, "__source_accessor__", None)
       if source_accessor is not None:
           return source_accessor(m, *args, **kwargs)
       return m.__nnsight_forward__(m, *args, **kwargs)
   ```
   This handles two responsibilities — `Envoy.skip` injection and source-accessor routing — without paying any cost when neither applies.

2. **Register a sentinel forward hook** that returns `output` unchanged:
   ```python
   module.register_forward_hook(lambda _, __, output: output)
   ```

   This sentinel is mandatory. PyTorch's `Module.__call__` fast-paths modules with zero hooks: if `len(_forward_hooks) == 0` at the time `__call__` is invoked, the dispatch path that fires hooks is skipped entirely. A one-shot hook that the worker registers just before the forward call would never fire. The sentinel guarantees PyTorch always takes the hook-dispatch path.

The sentinel is installed via `register_forward_hook` (the public API), so its handle is in `_forward_hooks` and the cleanup code in `add_ordered_hook` handles it correctly when other hooks are inserted in front of or behind it.

### The Mediator class

```python
class Mediator:
    intervention: Callable                    # the compiled function
    info: Tracer.Info                         # source, frame, line numbers
    name: str                                 # default f"Mediator{id(self)}"
    idx: int                                  # set by Interleaver.__enter__
    batch_group: Optional[List[int]]          # [start, size] or None for empty invokes

    interleaver: Interleaver                  # set on start()
    event_queue: Mediator.Value               # worker -> main
    response_queue: Mediator.Value            # main -> worker
    worker: Thread                            # the worker thread

    history: Set[str]                         # providers seen so far
    user_cache: List[Cache]
    hooks: List[Any]                          # every hook handle this mediator owns
    iteration_tracker: defaultdict(int)       # per-provider "current step"
    iteration: Optional[int]                  # explicit iter-loop target
    all_stop: Optional[int]                   # for `tracer.all()` bound

    transform: Optional[Callable]             # one-shot eproperty.transform
    cross_invoker: bool                       # whether to push/pull frame vars
```

Two state variables drive most of the complexity: `history` and `iteration_tracker`. `history` records every provider that has been seen *but not consumed* (so we can detect out-of-order requests); `iteration_tracker` records how many times a provider has fired (so we can suffix requesters correctly).

### Mediator.idx and ordering

`Mediator.idx` is set in `Interleaver.__enter__` from `enumerate(self.mediators)` (`interleaver.py:576`-`577`). It is the index of the invoke in definition order. Why it matters:

- `add_ordered_hook` reads `hook.mediator_idx` to decide where to insert the new hook in PyTorch's `_forward_hooks` dict. Lower idx fires earlier.
- This guarantees that if two invokes both hook the same module's output, invoke 0's hook runs before invoke 1's hook, matching the worker-thread serial execution order.
- The idx is also why `mediator.batch_group[0] == -1` is used as a "this mediator is not scheduled in this forward pass" sentinel by cache hooks (`hooks.py:383`); that case arises in vLLM where requests are added/removed across iterations.

### Mediator.handle — the event loop

```python
def handle(self, provider=None, value=None):
    # Save and restore current/value/provider so this is reentrant
    prev_current = self.interleaver.current
    prev_value = self.interleaver.batcher.current_value
    prev_provider = self.interleaver.batcher.current_provider

    self.interleaver.current = self
    self.interleaver.batcher.current_value = value
    self.interleaver.batcher.current_provider = provider

    process = self.event_queue.has_value
    while process:
        event, data = self.event_queue.get()
        if event == Events.VALUE:    process = self.handle_value_event(data, provider)
        elif event == Events.SWAP:   process = self.handle_swap_event(provider, *data)
        elif event == Events.EXCEPTION: process = self.handle_exception_event(data)
        elif event == Events.SKIP:   process = self.handle_skip_event(provider, *data)
        elif event == Events.BARRIER: process = self.handle_barrier_event(provider, data)
        elif event == Events.END:    process = self.handle_end_event()

    value = self.interleaver.batcher.current_value
    self.interleaver.current = prev_current
    self.interleaver.batcher.current_value = prev_value
    self.interleaver.batcher.current_provider = prev_provider
    return value
```

`process` is the loop condition. Each handler returns `True` if the worker has been responded to and the next event should be processed, or `False` if the worker is now blocked waiting for a different provider (and we should return to the model so it can proceed to the next module). `handle_value_event` (the most common case) returns `True` when the requested provider matched, or `False` (and re-stores the event) when it didn't.

The save/restore around `current` and `batcher.current_value` makes `handle()` reentrant — important because the BARRIER handler temporarily sets `current` to other mediators while broadcasting a barrier release.

### Per-event semantics

#### VALUE (`handle_value_event`, `interleaver.py:1013`)

- If `provider == requester`: take `batcher.narrow(self.batch_group)` to slice the batch, `respond(value)` to unblock the worker, and (if an eproperty registered a `transform` callback for this access) invoke the transform now and `batcher.swap` the result back into the model.
- If `requester` was already in `history`: respond with `OutOfOrderError`. This means the worker asked for a module that has already passed.
- Otherwise: add `provider` to `history` and re-store the event. Returns `False` so the model can proceed.

#### SWAP (`handle_swap_event`, `interleaver.py:1062`)

Same matching logic as VALUE but on a match it calls `batcher.swap(...)` to overwrite the live value, then `respond()` (no value).

#### SKIP (`handle_skip_event`, `interleaver.py:1154`)

On match, mutates `kwargs["__nnsight_skip__"] = value` directly into `batcher.current_value`'s kwargs dict. The `nnsight_forward` wrapper checks for this key and returns the value without calling the original forward. Replaces the old `SkipException`-based skip mechanism, which doesn't work with one-shot hooks because the exception would unwind through hooks the worker hasn't seen yet.

#### BARRIER (`handle_barrier_event`, `interleaver.py:1123`)

When a `Barrier` (`tracing/tracer.py:646`) fires, the *last* participating invoke calls `mediator.send(Events.BARRIER, participants_set)`. The `participants` set is non-None for that final invoker; the BARRIER handler iterates the interleaver's mediators, finds each participant, switches `interleaver.current`, calls `mediator.respond()` on each (releasing their workers), and then calls `mediator.handle(provider, value)` to drain any events those workers immediately produce. Earlier participants pass `None` and just block waiting.

#### END (`handle_end_event`, `interleaver.py:1146`)

The intervention finished cleanly. Cancels the mediator (clears worker, resets state) and returns `False`. The model continues executing without this mediator.

#### EXCEPTION (`handle_exception_event`, `interleaver.py:1100`)

The intervention raised. Cancels the mediator, then (unless the exception is `Cancelation`) calls `wrap_exception(e, self.info)` to rebuild the user-visible traceback and re-raises on the main thread. This is the failure path that ends up surfacing to the user's `try/except` around the trace.

### check_dangling_mediators

After the model finishes, `Envoy.interleave` calls `interleaver.check_dangling_mediators()` (`interleaver.py:652`). For every mediator still alive (i.e. blocked in `event_queue.wait()` waiting for a value):

- Pop its pending event.
- If its current iteration is 0, respond with `MissedProviderError("Execution complete but ... was not provided")`.
- Otherwise (iteration > 0), respond with the same error and warn — this is the common case for unbounded `tracer.iter[:]` where the loop body wants more steps than the model produces.

This is the source of the `OutOfOrderError` and `MissedProviderError` messages in the user-facing error catalog.

## Key files / classes

- `src/nnsight/intervention/interleaver.py:375` — `Interleaver`. Holds mediators, batcher, current; `handle()` fan-out.
- `src/nnsight/intervention/interleaver.py:481` — `Interleaver.wrap_module`. Skippable forward + sentinel hook.
- `src/nnsight/intervention/interleaver.py:567` — `Interleaver.__enter__`. Starts every mediator.
- `src/nnsight/intervention/interleaver.py:601` — `Interleaver.handle`. Provider-side fan-out to all mediators.
- `src/nnsight/intervention/interleaver.py:446` — `iterate_requester`. Iteration suffix resolution.
- `src/nnsight/intervention/interleaver.py:718` — `Mediator`. Per-invoke worker thread + event loop.
- `src/nnsight/intervention/interleaver.py:949` — `Mediator.handle`. Reentrant event dispatcher.
- `src/nnsight/intervention/interleaver.py:1013` — `handle_value_event`. The most-common case.
- `src/nnsight/intervention/interleaver.py:1100` — `handle_exception_event`. Re-raises with reconstructed traceback.
- `src/nnsight/intervention/interleaver.py:1123` — `handle_barrier_event`. Multi-invoke synchronization.
- `src/nnsight/intervention/interleaver.py:1154` — `handle_skip_event`. Injects `__nnsight_skip__` into kwargs.
- `src/nnsight/intervention/interleaver.py:1354` — `Mediator.remove_hooks`. Drains every hook handle.
- `src/nnsight/intervention/interleaver.py:338` — `Events` enum. The six event types.

## Lifecycle / sequence

For a single-invoke trace `with model.trace("hi"): hidden = model.layer.output.save()`:

1. `Interleaver.__enter__` runs, sets `_interleaving=True`, iterates mediators (just one).
2. `Mediator.start(interleaver)` saves the interleaver, captures the current CUDA stream, builds a `_worker_target` closure, starts the worker `Thread`, and waits on `event_queue.wait()`.
3. Worker thread runs the compiled intervention. `model.layer.output` triggers `eproperty.__get__`:
   - `requires_output` checks if `batcher.current_provider` already matches; it doesn't, so it calls `output_hook(mediator, self._module, "model.layer.output")` to install a one-shot forward hook with `mediator_idx = 0` at `iteration_tracker["model.layer.output"]` of 0.
   - The descriptor calls `interleaver.iterate_requester("model.layer.output") -> "model.layer.output.i0"`.
   - It calls `interleaver.current.request("model.layer.output.i0")`, which enqueues `(VALUE, "model.layer.output.i0")` and waits on `response_queue`.
4. `start()` returns. Main thread runs `model.interleave` -> `model.forward(...)`.
5. PyTorch fires module hooks. The sentinel hook fires first (idx=-inf), then the one-shot output hook fires:
   - It calls `mediator.handle("model.layer.output.i0", output)`.
   - `handle_value_event` matches; calls `batcher.narrow(batch_group)` and `respond(value)`.
   - `respond()` puts the value in `response_queue` and waits on `event_queue` (the worker is about to send another event).
6. Worker thread resumes from `request`. The `eproperty` `_preprocess` (if any) runs, then user code does `.save()`.
7. Eventually the worker reaches the `else: __nnsight_mediator__.end()` branch of the compiled function. `END` event lands in `event_queue`.
8. The hook returns; main thread continues forward; eventually `event_queue.wait()` in `respond()` (step 5) returns because the worker enqueued `END`.
9. Main thread processes `END` via `handle_end_event`, which cancels the mediator. Worker thread terminates.
10. Model finishes. `interleaver.check_cache_full()` runs (warns if cached modules were missed). `interleaver.check_dangling_mediators()` runs (no-op since mediator is dead).
11. `with self.interleaver:` exits; `Envoy.interleave`'s `finally` calls `interleaver.cancel()` to ensure idempotent cleanup.

## Extension points

- **A custom `Mediator` subclass.** `BackwardsMediator` (`tracing/backwards.py:69`) is the only example; it overrides `request()` to enforce that only `.grad` requests are allowed inside a backward context. You would do this for any tracer type that needs different validation on requester strings.
- **A new event type.** The `Events` enum has six types and the `Mediator.handle` switch dispatches on them. Adding a new event means: extend the enum, add a `Mediator.send_*` method, add a `handle_*_event` handler. This is rarely necessary; most needs can be expressed with VALUE/SWAP plus a custom requester naming convention.
- **A custom `iterate_requester`.** The base implementation handles the two-source resolution (explicit iter loop vs tracker). A subclass `Interleaver` could implement different iteration semantics — for example, a probabilistic or non-monotonic iteration policy for streaming runtimes.

## Related

- `docs/developing/lazy-hook-system.md` — the one-shot hook lifecycle that drives `mediator.handle()`.
- `docs/developing/eproperty-deep-dive.md` — how the worker arrives at `mediator.request()` in the first place.
- `docs/developing/source-accessor-internals.md` — how operation-level hooks plug into the same `mediator.handle()`.
- `docs/developing/batching-internals.md` — how `batcher.narrow` and `batcher.swap` slice the batch for each mediator.
- `NNsight.md` Section 3 — the original design narrative for the interleaver.

---
title: Interleaver and Controller
one_liner: One shared Interleaver installs a controller forward on every wrapped module; it passes through when idle and hands input/output to Interleaver.handle when interleaving, counting each visit once on the interleaver.
tags: [concept, mental-model, controller]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/envoy.md, docs/concepts/source-tracing.md]
sources: [src/nnsight/intervention/interleaver.py]
---

# Interleaver and Controller

## What this is for

How values get from the running model into a parked worker. One `Interleaver` (`interleaver.py`) is shared across an entire `Envoy` tree, so every module reports into the same set of workers.

How a module reaches it:

- **One controller per module, installed at wrap time.** When an `Envoy` is built it calls `interleaver.instrument(self)`, which installs the module's **controller** as its `forward` (`source.py`, `make_controller`). The controller is the whole per-module mechanism: it hands the input to `handle("{path}.input", ...)`, consults the `.skip` gate, runs the real forward, and hands the output to `handle("{path}.output", ...)`. No PyTorch forward hooks are registered — a module with none is called on PyTorch's fast path, which is most of what makes an instrumented model cheap.
- **Pass-through when idle.** The controller first checks that its interleaver is `interleaving` *and* `busy` (has workers); otherwise it runs the body and nothing else. So an instrumented model runs at engine speed when you're not tracing, and a vLLM step with no nnsight requests in it costs each module one attribute test.
- **Routed when interleaving.** Inside a trace, the handoffs return the handled value, so an intervention can edit input or output in place.
- **No hooks under tensor parallelism either.** transformers TP all-reduces and all-gathers in *its* forward hooks, which run after the controller — so the controller sees a row-parallel output as this rank's partial sum and a gathered head as a shard. `TPFragments` records which, and makes the value whole (all-reduce or all-gather) only when a worker is actually waiting there.
- **One occurrence counter per location**, on the interleaver, bumped once per visit; each worker holds the counts it started at, so its own occurrence of a location is a subtraction, not a per-worker increment at every handoff.

There is **one** primitive — a location string plus `Interleaver.handle` — and modules, source operations, `.skip`, and `result` all ride on it.

## When to use / when not to use

You never call these directly — `Envoy` properties do. Read this to:

- Understand why an untraced module is cheap (no hooks; the controller short-circuits on `interleaving`/`busy`).
- Debug `OutOfOrderError`.
- Plumb values into the interleaver from a custom driver (vLLM logits, generation results) via `Interleaver.handle`.

## Instrumenting a module

`Interleaver.instrument(envoy)`:

1. Lets `fragments.instrument(envoy)` record what this module's values are at the handoff (a shard or a partial sum, on a TP-split module).
2. Calls `install_controller(envoy)` (see [Source Tracing](source-tracing.md)) to install the controller forward and register this interleaver on the module under its path — so the module hands off, can be skipped or source-drilled, even when several envoys share it.

```python
def controller(*args, **kwargs):                      # the module's forward, per module
    interleaver, path, locations = state.active()     # which trace reaches this module now
    if interleaver is None or not interleaver.busy:   # no trace, or no workers: straight through
        return body(module, *args, **kwargs)
    args, kwargs = interleaver.handle(locations[0], (args, kwargs))   # "{path}.input"
    output = interleaver.handle(locations[1], NO_SKIP)                # the .skip gate
    if output is NO_SKIP:
        output = body(module, *args, **kwargs)
    return interleaver.handle(locations[2], output)                   # "{path}.output"
```

## Interleaver.handle: one call, every worker

`Interleaver.handle(provider, value)` (`interleaver.py`) is the whole model→worker interface:

1. Offer `value` to every mediator in turn via `Mediator.handle` (`interleaver.py`). A read is served the value; a swap replaces it. The value threads through all workers, so each sees the previous one's edits.
2. If batching, reassemble a batched `.skip` from its per-invoke parts (`Batcher.assemble_skip`).
3. Offer the now post-intervention value to any active caches (`tracer.cache()`), narrowed to each cache's own batch rows.
4. Return the possibly-edited value back to the controller, which substitutes it into the forward.

Values that don't come from a module use the same call: `Envoy.interleave` does `handle("result", result)`; source operations do `handle("{op}.input"/".output"/".skip", ...)`.

## Occurrence tagging (iteration)

A location can be reached many times in one run — every step of a generation loop revisits every module. `Mediator.handle` tracks this per location in `iterations`: this visit is the `iterations[provider]`-th, so it serves requests tagged `"{provider}.i{n}"` for that `n`, then increments.

- With **no `tracer.iter`**, a worker always parks with tag `.i0`, so every request binds to the **first** visit — the original single-forward behavior.
- Inside `tracer.iter[k]`, the worker pins its `iteration` to `k`; a request tagged `.i{k}` waits while earlier visits pass by, and binds on the k-th. After the first hit of a pinned non-zero step, the mediator *relaxes* (`iteration = None`) so the rest of that step's requests follow the model sequentially.

Because a **source operation** goes through `handle` every time it fires, an op inside a loop (an MoE expert loop, say) advances its own occurrence counter per *fire*, while a module advances once per forward — see [Source Tracing](source-tracing.md). No separate counter hooks are needed; it falls out of the one `handle` primitive.

## Out-of-order and dangling workers

After the model returns, `check_dangling_mediators` (`interleaver.py`) inspects any worker still parked:

- **`iteration == 0`** (a plain request the model ran past or never made): throw `OutOfOrderError` into the worker so the traceback points at the waiting line.
- **`iteration != 0`** (a `tracer.iter` loop — bounded or open — that outran the model's steps): throw to unwind the worker's `finally` blocks, but catch it and **warn** rather than raise — reached steps' saved values are kept.
- **`Event.BARRIER`** still pending: a barrier fewer blocks reached than it was built for — raise a `ValueError` pointing at the waiting line.

## Caches are post-intervention observers

`tracer.cache()` needs no per-module hooks. It registers a `Cache` on the calling mediator; `Interleaver.handle` feeds every location to every active cache *after* interventions have run, so a cache records exactly the values interventions produced, narrowed to that worker's rows. See `intervention/cache.py`.

## Skip is a gate

`.skip(value)` parks on `Event.SKIP` at `"{path}.skip"`. The module's **controller** forward (installed by `install_controller`) queries that gate *before* running the body: if a replacement is pending, the body is skipped and the replacement returned. This is why a skip can even read the module's own `.input` first — the controller is bound before `nn.Module.__call__` runs its pre-hooks. See [Source Tracing](source-tracing.md) for the controller.

## Gotchas

- **The controller is installed at wrap time and stays.** Overhead when idle is one frame and one `interleaving`/`busy` check per module call, on PyTorch's hook-free fast path. Don't replace a module's `forward` by hand — reassign the module through the envoy so `instrument` re-runs.
- **The input handoff sees `(args, kwargs)`** and can rewrite either; `.inputs` exposes the full pair, `.input` the first positional-or-keyword argument.
- **One interleaver per tree, reused across runs.** `cancel()` clears mediators and the batcher after each run; the controllers stay installed. A server keeps the same interleaver across requests.
- **Occurrence tags are strings.** A request pinned to a later step simply doesn't string-match earlier visits and waits — there is no numeric comparison in the hot path.

## Related

- [Threading and Mediators](threading-and-mediators.md) — the worker side that parks and is served here.
- [Envoy](envoy.md) — the properties that call `Mediator.value`/`swap`, which the controller's handoffs fulfil.
- [Source Tracing](source-tracing.md) — the same `handle` primitive applied to intra-forward operations, and the skip controller.
- Source: `src/nnsight/intervention/interleaver.py` (`Interleaver.instrument`, `Interleaver.handle`, `check_dangling_mediators`), `src/nnsight/intervention/cache.py` (`Cache`).

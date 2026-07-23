---
title: Interleaver and Hooks
one_liner: One shared Interleaver installs a pre-forward and a forward hook on every wrapped module; they pass through when idle and route input/output through Interleaver.handle when interleaving, tagging each visit with an occurrence index.
tags: [concept, mental-model, hooks]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/envoy.md, docs/concepts/source-tracing.md]
sources: [src/nnsight/intervention/interleaver.py:430, src/nnsight/intervention/interleaver.py:521, src/nnsight/intervention/interleaver.py:566, src/nnsight/intervention/interleaver.py:375, src/nnsight/intervention/interleaver.py:605]
---

# Interleaver and Hooks

## What this is for

How values get from the running model into a parked worker. One `Interleaver` (`interleaver.py:430`) is shared across an entire `Envoy` tree, so every module reports into the same set of workers.

The hook model in this rewrite:

- **Two hooks per module, installed at wrap time.** When an `Envoy` is built it calls `interleaver.instrument(self)` (`interleaver.py:521`), registering a `register_forward_pre_hook` and a `register_forward_hook` on the module.
- **Pass-through when idle.** Both hooks check `self.interleaving` first; outside a trace they return `None` and the module runs untouched. So an instrumented model runs at normal speed when you're not tracing.
- **Routed when interleaving.** Inside a trace, the pre-hook routes the module's `(args, kwargs)` through `handle("{path}.input", ...)` and the forward hook routes the output through `handle("{path}.output", ...)`. Because both hooks *return* the handled value, an intervention can edit input or output in place.

This is simpler than the older lazy one-shot hook design: there is **one** primitive — a location string plus `Interleaver.handle` — and modules, source operations, `.skip`, and `result` all ride on it.

## When to use / when not to use

You never call these directly — `Envoy` properties do. Read this to:

- Understand why an untraced module is cheap (its hooks short-circuit on `interleaving == False`).
- Debug `OutOfOrderError`.
- Plumb values into the interleaver from a custom driver (vLLM logits, generation results) via `Interleaver.handle`.

## Instrumenting a module

`Interleaver.instrument(envoy)` (`interleaver.py:521`):

1. Calls `install_skip(envoy)` (see [Source Tracing](source-tracing.md)) to install the per-module **controller** forward and register this interleaver on the module — so the module can be skipped or source-drilled, even when several envoys share it.
2. Removes any hooks previously installed for this path (`remove`), then registers the pre-forward and forward hooks (both `with_kwargs=True`).

```python
def pre_forward(module, args, kwargs):
    if not self.interleaving:
        return None
    return self.handle(f"{path}.input", (args, kwargs))   # editable input

def forward(module, args, kwargs, output):
    if not self.interleaving:
        return None
    return self.handle(f"{path}.output", output)          # editable output
```

`instrument` runs again on `Envoy._update` (dispatch swapping meta weights for real ones): it drops the old path's hooks and re-installs on the new module.

## Interleaver.handle: one call, every worker

`Interleaver.handle(provider, value)` (`interleaver.py:566`) is the whole model→worker interface:

1. Offer `value` to every mediator in turn via `Mediator.handle` (`interleaver.py:375`). A read is served the value; a swap replaces it. The value threads through all workers, so each sees the previous one's edits.
2. If batching, reassemble a batched `.skip` from its per-invoke parts (`Batcher.assemble_skip`).
3. Offer the now post-intervention value to any active caches (`tracer.cache()`), narrowed to each cache's own batch rows.
4. Return the possibly-edited value back to the hook, which substitutes it into the forward.

Values that don't come from a module hook use the same call: `Envoy.interleave` does `handle("result", result)`; source operations do `handle("{op}.input"/".output"/".skip", ...)`.

## Occurrence tagging (iteration)

A location can be reached many times in one run — every step of a generation loop revisits every module. `Mediator.handle` tracks this per location in `iterations`: this visit is the `iterations[provider]`-th, so it serves requests tagged `"{provider}.i{n}"` for that `n`, then increments.

- With **no `tracer.iter`**, a worker always parks with tag `.i0`, so every request binds to the **first** visit — the original single-forward behavior.
- Inside `tracer.iter[k]`, the worker pins its `iteration` to `k`; a request tagged `.i{k}` waits while earlier visits pass by, and binds on the k-th. After the first hit of a pinned non-zero step, the mediator *relaxes* (`iteration = None`) so the rest of that step's requests follow the model sequentially.

Because a **source operation** goes through `handle` every time it fires, an op inside a loop (an MoE expert loop, say) advances its own occurrence counter per *fire*, while a module advances once per forward — see [Source Tracing](source-tracing.md). No separate counter hooks are needed; it falls out of the one `handle` primitive.

## Out-of-order and dangling workers

After the model returns, `check_dangling_mediators` (`interleaver.py:605`) inspects any worker still parked:

- **`iteration == 0`** (a plain request the model ran past or never made): throw `OutOfOrderError` into the worker so the traceback points at the waiting line.
- **`iteration != 0`** (an open-ended `tracer.iter[:]` that outran the model's steps): throw to unwind the worker's `finally` blocks, but catch it and **warn** rather than raise — reached steps' saved values are kept.
- **`Event.BARRIER`** still pending: a barrier fewer blocks reached than it was built for — raise a `ValueError` pointing at the waiting line.

## Caches are post-intervention observers

`tracer.cache()` needs no per-module hooks. It registers a `Cache` on the calling mediator; `Interleaver.handle` feeds every location to every active cache *after* interventions have run, so a cache records exactly the values interventions produced, narrowed to that worker's rows. See `intervention/cache.py`.

## Skip is a gate, not a hook

`.skip(value)` parks on `Event.SKIP` at `"{path}.skip"`. The module's **controller** forward (installed by `install_skip`) queries that gate *before* running the body: if a replacement is pending, the body is skipped and the replacement returned. This is why a skip can even read the module's own `.input` first — the controller is bound before `nn.Module.__call__` runs its pre-hooks. See [Source Tracing](source-tracing.md) for the controller.

## Gotchas

- **Hooks are always installed, not lazy.** Overhead when idle is a single `if not self.interleaving: return None` per hook. Don't remove the module's hooks by hand — reassign the module through the envoy so `instrument` re-runs.
- **`with_kwargs=True` on both hooks.** The pre-hook sees `(args, kwargs)` and can rewrite either; `.inputs` exposes the full pair, `.input` the first positional-or-keyword argument.
- **One interleaver per tree, reused across runs.** `cancel()` clears mediators and the batcher after each run; the hooks stay installed. A server keeps the same interleaver across requests.
- **Occurrence tags are strings.** A request pinned to a later step simply doesn't string-match earlier visits and waits — there is no numeric comparison in the hot path.

## Related

- [Threading and Mediators](threading-and-mediators.md) — the worker side that parks and is served here.
- [Envoy](envoy.md) — the properties that call `Mediator.value`/`swap`, which the hooks fulfil.
- [Source Tracing](source-tracing.md) — the same `handle` primitive applied to intra-forward operations, and the skip controller.
- Source: `src/nnsight/intervention/interleaver.py` (`Interleaver.instrument`, `Interleaver.handle`, `check_dangling_mediators`), `src/nnsight/intervention/cache.py` (`Cache`).

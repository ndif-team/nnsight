---
title: Hook System
one_liner: One controller forward per module, installed at instrument time, carries the input/output handoff and the skip gate; no PyTorch hooks anywhere.
tags: [internals, dev]
related: [docs/developing/interleaver-internals.md, docs/developing/architecture-overview.md, docs/developing/source-internals.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/source.py, src/nnsight/intervention/envoy.py]
---

# Hook System

> **Renamed, twice over.** This page was `lazy-hook-system.md`, and then described
> two PyTorch hooks per module gated on a flag. Neither exists now: the module's
> **controller** (its replaced `forward`) carries the handoff itself, and no
> PyTorch hooks are registered.

## What this covers

How nnsight makes a module's forward pass observable and editable:

- The controller `Interleaver.instrument` installs as each module's `forward`, at
  construction time, which hands `.input`/`.output` to the interleaver, gates
  `.skip`, and (on demand) runs the source-instrumented body.
- Why it passes through when no trace is running, and why there are no PyTorch
  hooks on an ordinary module — the whole reason an instrumented model stays on
  PyTorch's fast call path.
- How tensor parallelism fits: the controller runs inside transformers' own
  hooks, so it sees the pre-collective value, which `TPFragments` knows how to
  make whole.

## Architecture

### The controller is installed at instrument time, not on demand

`Interleaver.instrument(envoy)` (`src/nnsight/intervention/interleaver.py:521`) runs
from `Envoy.__init__` (`envoy.py:135`) for every module in the tree, and again from
`Envoy._update` (`envoy.py:257`) when meta weights are swapped for real ones. It
installs one controller as the module's `forward` (`install_controller` →
`_make_controller`, `source.py`):

```python
def controller(*args, **kwargs):
    interleaver, path, locations = state.active()     # the trace reaching this module now
    if interleaver is None or not interleaver.busy:   # no trace, or no workers
        return body(module, *args, **kwargs)
    args, kwargs = interleaver.handle(locations[0], (args, kwargs))   # "{path}.input"
    output = interleaver.handle(locations[1], NO_SKIP)                # the .skip gate
    if output is NO_SKIP:
        output = body(module, *args, **kwargs)
    return interleaver.handle(locations[2], output)                   # "{path}.output"
```

The handoffs see and can rewrite the full `(args, kwargs)` / `output`. Because
each **returns** the handled value, an intervention can edit the module's input or
output in place. There are no PyTorch hooks at all, tensor parallelism included:
the controller runs inside transformers' own hooks, so `TPFragments` describes
the value at that point (a shard, or a partial sum its post-hook will reduce) and
makes it whole only when a worker is waiting.

### Pass-through when idle

The first line of each hook is the whole "laziness" story:
`if not self.interleaving: return None`. Outside a trace — and there is exactly one
`interleaving` flag per interleaver, flipped in `Interleaver.__enter__`/`__exit__`
(`:489`/`:509`) — the hooks return `None`, which tells PyTorch "no change," and the
forward pass runs exactly as if the hooks weren't there. There is a constant
two-hook dispatch cost per module even when idle, but no Python-level work and no
per-mediator loop. Inside a trace, `handle` (`:566`) fans the value out to every
worker parked on that location.

This is why there is no sentinel-hook trick: the hooks are permanent, so PyTorch's
"zero hooks → skip dispatch" fast path never applies. The OLD design installed
hooks on demand mid-forward and needed a sentinel to keep the dispatch path live;
that entire problem is gone.

### No ordering machinery

Hook firing order across *modules* is just the model's execution order. Ordering
across *workers* for the same location is the order of `interleaver.mediators`
(definition order), applied inside the single `handle` call's loop — see
`docs/developing/interleaver-internals.md`. There is no `mediator_idx`, no
`add_ordered_hook`, and no rebuilding of PyTorch's internal hook dicts.

### Caches are not hooks

`tracer.cache(...)` does not install its own PyTorch hooks. A `Cache` is registered
on the calling worker's mediator (`mediator.caches`, set in
`InterleavingTracer.cache`, `tracer.py:201`), and `Interleaver.handle` feeds every
location's post-intervention value to each active cache after serving the workers
(`interleaver.py:593`). So a cache observes exactly what the controllers
already surface — no extra registration, and it always sees post-intervention
values scoped to its worker's batch rows.

## The source/skip controller

The controller surfaces `.input` and `.output`, gates `.skip`, and on demand runs
the source-instrumented body for operation-level access. It is installed by
`install_controller(envoy)` (`source.py:437`), called from `instrument` up front so the
gate is in place before `nn.Module.__call__` binds `forward`.

### `_State` — per-module, shared across interleavers

`_State` (`source.py:80`) is stored at `module.__dict__["__nnsight__"]`. It holds:

- `interleavers` — a `weakref.WeakKeyDictionary` mapping each interleaver that
  instrumented this module to the path it addresses the module by. `active()`
  (`source.py:113`) picks the one whose trace is currently running (`interleaving`
  is `True`) — at most one, exactly as each stacked module's hooks fire only under
  their own interleaver.
- `body` — the *unbound* original forward (or the source-instrumented one once
  `.source` is used).
- `sourced` — whether `body` is the instrumented forward.

Weak keys and an unbound `body` keep the state from pinning the module (which owns
the state — a cycle), so a finished local trace's interleaver drops out on its own
while a server's persistent interleaver stays and serves request after request.

### The controller

`_make_controller` (`source.py:362`) builds the forward installed on the module:

```python
def controller(*args, **kwargs):
    module = module_ref()
    state = module.__dict__[_STATE]
    interleaver, path = state.active()
    if interleaver is None:                      # no trace running: run normally
        return state.body(module, *args, **kwargs)
    skipped = _skipped(interleaver, path)        # offer the ".skip" gate
    if skipped is not _NO_SKIP:
        return skipped                           # a worker skipped this module
    return state.body(module, *args, **kwargs)
```

`_skipped` (`source.py:266`) is just `interleaver.handle(f"{path}.skip", _NO_SKIP)`:
the skip gate is an ordinary `handle` on a `.skip` location, served by a worker's
`Event.SKIP` (from `envoy.skip(replacement)` → `Mediator.skip`). If no worker is
parked on `.skip`, `handle` returns the `_NO_SKIP` sentinel unchanged and the body
runs. This is why `envoy.skip(...)` can read the module's own input first: the input
is offered before the skip gate.

This replaces the OLD `__nnsight_skip__`-kwarg trick and the even older
`SkipException`. There is no exception unwinding and no magic kwarg — a skip is just
another location on the same `handle` primitive.

### Source instrumentation (on demand)

The first time a worker uses `envoy.source`, `install_source` (`source.py:414`)
upgrades `state.body` to a source-instrumented copy of the forward — one that
rewrites each call `fn(*a, **k)` into `__nnsight_op__("source.{name}_{n}", fn, *a,
**k)`, bracketing every operation with `.input`/`.output`/`.skip` locations. The
controller is unchanged; only its `body` swaps. Details are in
`docs/developing/source-internals.md`.

## Key files / classes

- `src/nnsight/intervention/interleaver.py:521` — `Interleaver.instrument`. Installs the controller.
- `:566` — `Interleaver.handle`. Fan-out that the controller calls.
- `:489`/`:509` — `interleaving` flag (`__enter__`/`__exit__`).
- `src/nnsight/intervention/source.py:437` — `install_controller`. Registers the controller.
- `:80` — `_State`. Per-module state; weakly-held interleaver routes.
- `:362` — `_make_controller`. The installed forward: input handoff, skip gate, body, output handoff.
- `:266` — `_skipped`. The `.skip` gate as a `handle` call.
- `:414` — `install_source`. Upgrades `body` to the instrumented forward.

## Lifecycle / sequence

For `with model.trace("hi"): model.layer.skip(model.layer.input)`:

1. At `Envoy` construction, `instrument` put the controller on `model.layer`'s
   `forward`. Inert (`interleaving` is `False`).
2. The trace enters; `interleaving` flips `True`; the worker starts.
3. The worker reads `model.layer.input` → parks on `model.layer.input.i0`.
4. `nn.Module.__call__` on `model.layer` calls the controller, whose
   `handle("layer.input", (args, kwargs))` serves the input to the worker; the
   worker computes `replacement` and calls `model.layer.skip(replacement)` →
   parks on `model.layer.skip.i0`.
5. The controller's `_skipped` → `handle("layer.skip", _NO_SKIP)`; the worker's
   `SKIP` event substitutes `replacement`; `body` is not run.
6. The controller still hands off the returned value (`handle("layer.output",
   replacement)`), so downstream reads of `model.layer.output` see the skip result.

## Extension points

- **A new gated location.** Anything served by `handle(location, value)` and read/
  written by an `Envoy` property works — you don't add hooks, you add locations.
- **A hook on a non-standard module.** If a module bypasses PyTorch's forward
  dispatch, `instrument`'s hooks won't fire; expose its values by having the driver
  call `interleaver.handle(location, value)` directly (as vLLM does).

## Related

- `docs/developing/interleaver-internals.md` — what `handle` does after a hook fires.
- `docs/developing/source-internals.md` — the instrumented-forward path.
- `docs/concepts/interleaver-and-hooks.md` — the mental-model version.

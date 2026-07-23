---
title: Hook System
one_liner: Forward pre/forward hooks installed once per module at instrument time, pass-through when idle; plus the per-module source/skip controller.
tags: [internals, dev]
related: [docs/developing/interleaver-internals.md, docs/developing/architecture-overview.md, docs/developing/source-internals.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/source.py, src/nnsight/intervention/envoy.py]
---

# Hook System

> **Renamed.** This page was `lazy-hook-system.md`. The "lazy, one-shot,
> mediator-ordered hook" design it described does not exist in this codebase — no
> `hooks.py`, no `add_ordered_hook`, no per-mediator hook lists, no sentinel hook.
> The current system installs two ordinary PyTorch hooks per module *once* and
> gates them on a flag. The page is rewritten to match.

## What this covers

How nnsight makes a module's forward pass observable and editable:

- The two forward hooks `Interleaver.instrument` installs per module, at
  construction time, and why they pass through when no trace is running.
- The source/skip controller that replaces a module's `forward` to add the `.skip`
  gate and (on demand) operation-level access.
- Why there is no fast-path or sentinel problem to work around.

## Architecture

### Hooks are installed at instrument time, not on demand

`Interleaver.instrument(envoy)` (`src/nnsight/intervention/interleaver.py:521`) runs
from `Envoy.__init__` (`envoy.py:135`) for every module in the tree, and again from
`Envoy._update` (`envoy.py:257`) when meta weights are swapped for real ones. It
registers exactly two hooks per module:

```python
def pre_forward(module, args, kwargs):
    if not self.interleaving:
        return None
    return self.handle(f"{path}.input", (args, kwargs))

def forward(module, args, kwargs, output):
    if not self.interleaving:
        return None
    return self.handle(f"{path}.output", output)

self.handles[path] = [
    module.register_forward_pre_hook(pre_forward, with_kwargs=True),
    module.register_forward_hook(forward, with_kwargs=True),
]
```

Both are registered with `with_kwargs=True`, so they receive and can rewrite the
full `(args, kwargs)` / `output`. Because each hook **returns** the handled value,
an intervention can edit the module's input (return `(args, kwargs)`) or output
(return a value) in place.

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
(`interleaver.py:593`). So a cache observes exactly what the existing forward hooks
already surface — no extra registration, and it always sees post-intervention
values scoped to its worker's batch rows.

## The source/skip controller

The forward hooks only surface two locations per module: `.input` and `.output`.
Everything else — the `.skip` gate and operation-level access — is added by
replacing the module's `forward` with a controller. This is installed by
`install_skip(envoy)` (`source.py:437`), called from `instrument` up front so the
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
is offered by the pre-hook before the controller's skip gate.

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

- `src/nnsight/intervention/interleaver.py:521` — `Interleaver.instrument`. Installs the two hooks + skip controller.
- `:566` — `Interleaver.handle`. Fan-out that the hooks call.
- `:489`/`:509` — `interleaving` flag (`__enter__`/`__exit__`).
- `:663`/`:668` — `remove`/`clear`. Drop hooks per path / all.
- `src/nnsight/intervention/source.py:437` — `install_skip`. Registers the controller.
- `:80` — `_State`. Per-module state; weak interleaver keys.
- `:362` — `_make_controller`. The installed forward: skip gate then body.
- `:266` — `_skipped`. The `.skip` gate as a `handle` call.
- `:414` — `install_source`. Upgrades `body` to the instrumented forward.

## Lifecycle / sequence

For `with model.trace("hi"): model.layer.skip(model.layer.input)`:

1. At `Envoy` construction, `instrument` installed the pre/forward hooks and
   `install_skip` put the controller on `model.layer`'s `forward`. All inert
   (`interleaving` is `False`).
2. The trace enters; `interleaving` flips `True`; the worker starts.
3. The worker reads `model.layer.input` → parks on `model.layer.input.i0`.
4. `nn.Module.__call__` on `model.layer` runs the pre-hook → `handle("layer.input",
   (args, kwargs))` serves the input to the worker; the worker computes
   `replacement` and calls `model.layer.skip(replacement)` → parks on
   `model.layer.skip.i0`.
5. `nn.Module.__call__` binds and calls the controller, which calls
   `_skipped` → `handle("layer.skip", _NO_SKIP)`; the worker's `SKIP` event
   substitutes `replacement`; the controller returns it without running `body`.
6. The forward hook still fires on the returned value (`handle("layer.output",
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

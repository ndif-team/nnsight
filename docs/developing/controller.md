---
title: The Controller
one_liner: One controller forward per module, installed when the envoy tree is built, carries the input/output handoff, the skip gate and the source-instrumented body; no `nn.Module` hooks are registered.
tags: [internals, dev]
related: [docs/developing/interleaver-internals.md, docs/developing/architecture-overview.md, docs/developing/source-internals.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py]
---

# The Controller

## What this covers

How nnsight makes a module's forward pass observable and editable: every wrapped
module's `forward` is a **controller** that hands the module's input and output to
the interleaver, offers a `.skip` gate, and runs the real body — or, once `.source`
has been used, an instrumented copy of it. No `nn.Module` hooks are registered, on any
runtime.

## The controller

`Interleaver.instrument(envoy)` runs from `Envoy.__init__` for every module in the
tree, and again from `Envoy._update` when meta weights are swapped for real ones. It
lets the runtime's `Fragments` record what the module's values are at the handoff,
then calls `install_controller(envoy)` (`source.py`), which installs one controller
as the module's `forward` (a `Controller`, whose `__call__` is):

```python
def __call__(self, *args, **kwargs):
    interleaver, path, locations = state.active()     # the trace reaching this module now
    if interleaver is None:                           # no trace, or no workers
        return body(module, *args, **kwargs)
    args, kwargs = interleaver.handle(locations[0], (args, kwargs))   # "{path}.input"
    output = interleaver.handle(locations[1], NO_SKIP)                # "{path}.skip"
    if output is NO_SKIP:
        output = body(module, *args, **kwargs)
    return interleaver.handle(locations[2], output)                   # "{path}.output"
```

Each handoff **returns** the handled value, so an intervention can rewrite the
module's input (`(args, kwargs)`) or its output, in place or by replacement. The
controller is stored in the module's instance `__dict__` under `forward`, so
`nn.Module.__call__` finds it ahead of the class's `forward`; it holds the module by
weakref (the module owns it) and keeps the original's signature with
`functools.update_wrapper`, which `generate()` introspects.

Copying or pickling a wrapped module rebinds its controller to the copy:
`Controller.__deepcopy__` looks the copied module up in `deepcopy`'s memo (the copy
is registered there before its `__dict__` is copied) and binds to it, and
`__reduce__` pickles it as the module and its `State`. A copy therefore computes
with its own weights, never the original's.

Being the forward rather than a hook keeps the module on PyTorch's fast call path.
It also means the controller runs *inside* whatever hooks the runtime itself
registers: under transformers tensor parallelism the collectives live in those
hooks, so the controller sees a row-parallel output as this rank's partial sum and
a column-parallel one as a shard — which is what `TPFragments` describes and makes
whole when a worker is waiting ([tensor-parallel.md](../models/tensor-parallel.md)).

### The body

`install_controller` takes the body from the module's own `forward` if it has one
(`module_body`), not from its class. The controller claims the *instance* slot, so
a forward already there — `self.forward = self._fast` picked in `__init__`, a
monkeypatched layer, `torch.compile`'s `OptimizedModule` — is one it would
otherwise destroy for the rest of the process. Such a forward is already bound, so
it is wrapped to take (and ignore) the module the controller passes.

Two wrappers are not bodies: accelerate's device-alignment wrapper, which
`run_body` re-applies from `_hf_hook`, and transformers' tensor-parallel wrapper,
which `_keep_tp_forward` rebuilds *around* the controller — running either as the
body would apply its transforms twice.

The traffic goes the other way too: anything that assigns `module.forward` *after*
nnsight wrapped it takes the controller's slot and silently disables every handoff
that module makes. The next `install_controller` on it warns
([out-of-order-error.md](../errors/out-of-order-error.md)).

### Pass-through when idle

`state.active()` returns an interleaver only if it is `interleaving` (inside
`__enter__`/`__exit__`) **and** `busy` (has workers, or is recording CUDA graphs).
Otherwise the controller calls the body and returns: one frame and one check per
module call, no dict lookups, no per-worker loop. A vLLM step that carries no
nnsight request costs the model exactly that.

### `State` — per-module, shared across interleavers

`State` lives at `module.__dict__["__nnsight__"]` and holds:

- `routes` — `(weakref(interleaver), path, locations)` for each interleaver that
  instrumented this module, at most one entry per interleaver. `active()` walks it
  and returns the first whose trace is running. A module wrapped by two trees routes
  to whichever is tracing; a finished local trace's interleaver drops out on its
  own, while a server's persistent interleaver stays and serves request after
  request.
- `body` — the *unbound* original forward or, once sourced, the instrumented one.
- `original` — what `body` was before instrumentation, which is what a copy or a
  pickle of the module carries (`__getstate__`): the instrumented body is built at
  run time, so pickle cannot name it and its `Compiled` holds a code object pickle
  cannot write at all. A copy re-instruments the first time it is sourced.
- `sourced` — whether `body` is the instrumented forward.

Weak references and an unbound `body` keep the state from pinning the module it
lives on. `__getstate__` drops the routes as well: their weakrefs are unpicklable,
and a copy belongs to whatever wraps it next, not to the traces the original is in.

### The skip gate

`.skip(replacement)` parks the worker on `Event.SKIP` at `"{path}.skip"`. The
controller offers that gate between the input handoff and the body: if a worker is
parked there, `handle` returns its replacement, the body is not run, and the
replacement still goes through the output handoff so downstream reads of
`.output` see it. Because the input is offered first, a skip's replacement can be
computed from the module's own input.

### Source instrumentation, on demand

The first time a block uses `envoy.source`, `install_source` upgrades `state.body`
to a source-instrumented copy of the forward, in which every call `fn(*a, **k)`
becomes `__nnsight_op__("source.{name}_{n}", fn, *a, **k)` — the same three
handoffs (`.input` / `.skip` / `.output`) one level down. The controller is
unchanged; only its `body` swaps. See
[source-internals.md](source-internals.md).

Caches (`tracer.cache(...)`) observe the values the controllers hand to
`Interleaver.handle`; see [interleaver-internals.md](interleaver-internals.md#caches).

## Lifecycle / sequence

For `with model.trace("hi"): model.layer.skip(model.layer.input)`:

1. At `Envoy` construction, `instrument` put the controller on `model.layer`'s
   `forward`. Inert: no interleaver is interleaving.
2. The trace enters; the worker starts and reads `model.layer.input` → parks on
   `model.layer.input`, occurrence 0.
3. `nn.Module.__call__` on `model.layer` calls the controller, whose
   `handle("model.layer.input", (args, kwargs))` serves the input to the worker; the
   worker computes `replacement` and calls `model.layer.skip(replacement)` → parks
   on `model.layer.skip`.
4. The controller's `handle("model.layer.skip", NO_SKIP)` finds the worker parked
   there; the `SKIP` event substitutes `replacement`; `body` is not run.
5. The controller hands the returned value through `handle("model.layer.output",
   replacement)`, so downstream reads see the skip result.

A module a runtime calls without going through `forward`, and new served
locations, are covered in
[interleaver-internals.md](interleaver-internals.md#extension-points) and
[extending-envoy.md](extending-envoy.md).

## Related

- [interleaver-internals.md](interleaver-internals.md) — what `handle` does with the value.
- [source-internals.md](source-internals.md) — the instrumented-forward path.
- [docs/concepts/interleaver-and-controller.md](../concepts/interleaver-and-controller.md) — the mental-model version.

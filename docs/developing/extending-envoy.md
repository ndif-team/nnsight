---
title: Exposing New Values on an Envoy
one_liner: The extension surface for a new hookable value — an eproperty descriptor whose stub is the read-side preprocess, served from the driver with .provide.
tags: [internals, dev]
related: [docs/developing/interleaver-internals.md, docs/developing/source-internals.md, docs/developing/adding-a-new-runtime.md]
sources: [src/nnsight/intervention/eproperty.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/interleaver.py, src/nnsight/modeling/vllm/vllm.py]
---

# Exposing New Values on an Envoy

> **This page documents the `eproperty` descriptor.** `.input`, `.inputs`,
> `.output`, and the runtime-internal values a runtime adds (`logits`, `samples`,
> a custom telemetry read) are all `eproperty` descriptors
> (`intervention/eproperty.py`); `.skip` is a method and `.source` a plain
> property. This is the surface for exposing a *new* hookable value.

## The whole mechanism

The interleaver exposes exactly one primitive: a **location string** and
`Interleaver.handle(location, value)`, which offers a produced value to every parked
intervention at that location and returns whatever an intervention wrote back. A
location is any string — a module path suffix like `"model.h.0.output"`, the run's
`"result"`, or a runtime value like `"model.logits"`.

An `eproperty` is the reusable read/write descriptor over one such location. It is a
small subclass of `property` (`eproperty.py:44`) with two ends:

- **Read side (the API a user writes).** Reading the attribute runs
  `Mediator.value(location)` (parking the worker until the model reaches it), then
  passes the served value through the descriptor's **preprocess** and hands you the
  result (`__get__`, `eproperty.py:101`). Writing runs an optional **postprocess**
  and then `Mediator.swap(location, value)` (`__set__`, `eproperty.py:115`).
- **Produce side (where the value exists).** `eproperty.provide(obj, value)`
  (`eproperty.py:120`) calls `obj.interleaver.handle(location, value)` — serving the
  value to a parked worker and returning it, edited if the worker wrote back.

The location is `"{obj.path}.{key}"`, or just `key` when the host has no `path` (as
for the tracer's `result`). A host is anything satisfying the `IEnvoy` protocol
(`eproperty.py`): it exposes an `interleaver` and an optional `path` — `path` is read
via `getattr(obj, "path", "")`, so a tracer host that omits it falls back to the bare
`key`. `Mediator.value` / `Mediator.swap` park the intervention greenlet until the
interleaver reaches that location, then hand back the value (or substitute one).
Reading then swapping the same location in one trace works — both events drain in a
single `handle`.

## Defining an eproperty

Decorate a stub with `@eproperty` (bare) or `@eproperty(key=..., description=...)`.
**The decorated stub *is* the preprocess** — it takes the raw value the interleaver
served and returns what the user reads, so an identity view is just `return value`:

```python
from nnsight.intervention.eproperty import eproperty

class MyModel(NNsight):
    @eproperty                          # key defaults to "telemetry"
    def telemetry(self, value):         # preprocess: served value -> what you read
        return value
```

- `key` — the location suffix (`"{path}.{key}"`); defaults to the stub's name.
  Several eproperties may share a key to give different views of one location —
  `inputs` uses `@eproperty(key="input")` so it shares `input`'s location.
- `description` — a short label. It has no effect on reads; it only surfaces the
  attribute in the Envoy repr tree as `(name): description`. `.input`/`.output`
  carry none, so they stay hidden; a runtime's `.logits` carries one, so it shows up.

Two more callbacks refine the descriptor:

- `@name.postprocess` (`eproperty.py:87`) — runs on a **written** value before it's
  swapped in. `Envoy.input` uses it to repack a lone first argument back into the
  full `(args, kwargs)` the model expects.
- `@name.transform` (`eproperty.py:92`) — the write-back half of a *reshaping*
  preprocess. When the preprocess returns a reshaped/sliced view, in-place edits to
  it are invisible to the model (which still holds the original); the transform maps
  the edited view back to the model's layout and fires once, after the block is done
  with the read, splicing the result in like a swap. `eproperty.py`'s module
  docstring carries the canonical per-head example.

## How `.output` already works

`Envoy.output` (`envoy.py:448`) is an identity-preprocess eproperty:

```python
@eproperty
def output(self, value: Any) -> Object:
    return value
```

Its location is `"{self.path}.output"`. The produce side is the forward hook
installed by `Interleaver.instrument` (`interleaver.py:521`), which calls
`handle(f"{path}.output", output)` after the module runs. `.input`, `.inputs`, and
`.skip` follow the identical pattern one location over. `SourceEnvoy`'s op-level
`.output`/`.input`/`.inputs` (`source.py:550`) are the same descriptors, keyed on an
operation's path.

## The canonical non-module example: `tracer.result`

The model's own return value is not a module output, yet you can read it. It's an
eproperty on the tracer (`tracer.py:106`) with no `description` and — because the
tracer has no `path` — a bare `"result"` location:

```python
class InterleavingTracer(Tracer):
    @eproperty
    def result(self, value):
        return value
```

`Envoy.interleave` (`envoy.py:649`) serves it under that location after the forward:

```python
with self.interleaver:
    result = fn(*args, **kwargs)
    self.interleaver.handle("result", result)   # serve the run's return value
```

That's the entire pattern for a value that isn't a module input/output: an
eproperty, and a `handle` (or `.provide`) where the value is produced.

## Adding a runtime-internal value (the vLLM `logits`/`samples` pattern)

vLLM surfaces the logits and sampled tokens — engine-internal values, not module
outputs — as eproperties on the model class, each given a `description` so they show
in the repr (`vllm.py:144`):

```python
class VLLM(Remotable):
    @eproperty(description="pre-sampling logits for this step")
    def logits(self, value):
        return value

    @eproperty(description="token ids drawn from logits this step")
    def samples(self, value):
        return value
```

The produce side lives where the engine computes those values — in the model runner,
inside an open interleaver context — and uses the eproperty's own `.provide` so the
two sides can't drift out of sync (`GPUModelRunner.py:450`, `:472`):

```python
# logits phase
logits = type(model).logits.provide(model, original)

# sampling phase
sampler_output.sampled_token_ids = type(model).samples.provide(
    model, sampler_output.sampled_token_ids
)
```

`.provide(model, value)` forwards to `model.interleaver.handle("model.logits", value)`
(the location the read side parks on), returns the value edited if an intervention
wrote to it. So a user writes `logits = model.logits.save()` inside a trace, and the
runner's `provide` serves it. No registration table — a descriptor and a `.provide`.

## Wiring constraint

An `eproperty` only works on a class that is actually used as an envoy — the base
`Envoy`, an `NNsight`/model subclass, `SourceEnvoy`, the tracer, or a runtime model
like `VLLM`. Child envoys default to the base `Envoy`, so to give a specific
submodule a subclass bearing a custom eproperty, pass `envoys=` — a map from a
module type or dotted path suffix to an `Envoy` subclass (`_resolve_envoy_class`,
`envoy.py`). Without `envoys=`, put the custom eproperty on the model/runtime
subclass (or the tracer).

## Recipe

1. **Add an eproperty to the model/runtime class (or the tracer).** Decorate a stub
   that takes `(self, value)` and returns what the user should read (`return value`
   for an identity view). Give it a `description` if it should appear in the repr; a
   `key=` only if it must share another location.
2. **Serve it where produced.** Wherever the value is computed, inside an open
   interleaver context, call `type(obj).<name>.provide(obj, value)` and use the
   return (an intervention may have edited it). Equivalently, `handle(location, ...)`
   directly.
3. **(Optional) refine it.** Add a `@name.postprocess` if writes need repacking, or
   a `@name.transform` if the preprocess returns a reshaped view users will edit.

That's the full extension surface. If the value is an operation *inside* a forward
(not a whole-module or engine value), you don't add anything — `.source` already
makes every call site addressable (see [source-internals.md](./source-internals.md)).

## Key files

- `src/nnsight/intervention/eproperty.py` — the `eproperty` descriptor: preprocess
  (`:78`), `postprocess` (`:87`), `transform` (`:92`), `__get__`/`__set__` (`:101`/`:115`), `provide` (`:120`)
- `src/nnsight/intervention/envoy.py` — `.input`/`.inputs`/`.output` eproperties (`:419`–`455`); `interleave` serving `"result"` (`:649`); the repr surfacing of described eproperties (`:897`)
- `src/nnsight/intervention/tracer.py:106` — `result` eproperty
- `src/nnsight/intervention/interleaver.py` — `Mediator.value`/`swap`/`skip`, `Interleaver.handle`, `Interleaver.instrument` (`:521`)
- `src/nnsight/modeling/vllm/vllm.py:144` — `logits`/`samples` eproperties (a real runtime example)

## Related

- [interleaver-internals.md](./interleaver-internals.md) — `handle` and the event protocol in depth
- [source-internals.md](./source-internals.md) — operation-level locations
- [adding-a-new-runtime.md](./adding-a-new-runtime.md) — where a runtime plugs its values in

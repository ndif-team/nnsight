---
title: Exposing New Values on an Envoy
one_liner: The extension surface for a new served value — an eproperty descriptor whose stub is the read-side preprocess, served from the driver with .provide.
tags: [internals, dev]
related: [docs/developing/interleaver-internals.md, docs/developing/source-internals.md, docs/developing/adding-a-new-runtime.md]
sources: [src/nnsight/intervention/eproperty.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/interleaver.py, src/nnsight/modeling/vllm/vllm.py]
---

# Exposing New Values on an Envoy

> **This page documents the `eproperty` descriptor.** `.input`, `.inputs`,
> `.output`, and the runtime-internal values a runtime adds (`logits`, `samples`,
> a custom telemetry read) are all `eproperty` descriptors
> (`intervention/eproperty.py`); `.skip` is a method and `.source` a plain
> property. This is the surface for exposing a *new* served value.

## The whole mechanism

The interleaver exposes exactly one primitive: a **location string** and
`Interleaver.handle(location, value)`, which offers a produced value to every parked
intervention at that location and returns whatever an intervention wrote back. A
location is any string — a module path suffix like `"model.h.0.output"`, the run's
`"result"`, or a runtime value like `"model.logits"`.

An `eproperty` is the reusable read/write descriptor over one such location — a
small subclass of `property` (`src/nnsight/intervention/eproperty.py`) with two
ends:

- **Read side (the API a user writes).** Reading the attribute runs
  `Mediator.value(location)` (parking the worker until the model reaches it), then
  passes the served value through the descriptor's **preprocess** and hands you the
  result (`__get__`). Writing runs an optional **postprocess** and then
  `Mediator.swap(location, value)` (`__set__`).
- **Produce side (where the value exists).** `eproperty.provide(obj, value)` calls
  `obj.interleaver.handle(location, value)` — serving the value to a parked worker
  and returning it, edited if the worker wrote back.

The location is `"{obj.path}.{key}"`, or just `key` when the host has no `path` (as
for the tracer's `result`). A host is anything satisfying the `IEnvoy` protocol: it
exposes an `interleaver` and an optional `path` — `path` is read
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
  attribute in the Envoy repr tree as `(name): description`, with the stub's return
  annotation after an arrow when it has one (`(logits) -> Logits: ...`).
  `.input`/`.output` carry none, so they stay hidden; a runtime's `.logits` carries
  one, so it shows up.

**What the raw value is depends on the key.** A `key="output"` eproperty is served
the module's output. A `key="input"` one is served the raw `(args, kwargs)` pair,
not a bare tensor — so a preprocess sharing `input`'s location has to destructure
it, and a `transform` has to repack it:

```python
@eproperty(key="input")
def heads(self, value):
    (x,), _ = value                         # the raw pair, not a tensor
    b, s, h = x.shape
    return x.view(b, s, self.n_heads, h // self.n_heads).transpose(1, 2)

@heads.transform
def heads(self, value, raw):
    b, nh, s, hd = value.shape
    return ((value.transpose(1, 2).reshape(b, s, nh * hd),), {})   # repack
```

**A preprocess that raises `AttributeError` is swallowed.** An eproperty is a
`property`, and a property getter raising `AttributeError` falls through to
`__getattr__` — so a typo inside your preprocess surfaces as
`'X' object (nor its module) has attribute 'y'`, naming the eproperty rather than
the line that failed. Raise something else, or catch and re-raise, if the
preprocess can legitimately fail.

Two more callbacks refine the descriptor:

- `@name.postprocess` (`eproperty.py`) — runs on a **written** value before it's
  swapped in. `Envoy.input` uses it to repack a lone first argument back into the
  full `(args, kwargs)` the model expects.
- `@name.transform` (`eproperty.py`) — the write-back half of a *reshaping*
  preprocess. A preprocess returning a `.view()` of the served tensor shares its
  storage, so in-place edits reach the model with no transform. One returning a
  copy (a `.reshape()` that cannot view, a stack, an arithmetic result) hides edits
  from the model, which still holds the original — and a `key="input"` preprocess
  is always in this case, since the served value is a container that something has
  to rebuild. The transform maps the edited view back to the model's layout. The read binds a `WriteBack`
  (`eproperty.py`) on the mediator, a swap the worker has not issued yet;
  `Mediator.flush` issues it when the worker next moves on (its next request, in
  `Mediator.event`, or the end of its block), tagged with the occurrence the view
  was read at. Until then a repeated read of the value is answered with the view it
  holds, so a statement that reads twice edits one tensor. It takes
  `(self, view, raw)`,
  the raw being the value as served: that is how a view that is a copy of one
  element of a tuple gets the rest of the tuple back, since a transform cannot read
  the location itself, firing on the model side after the read. A transform must
  return a value of the *location's* shape: on a tuple-returning module it returns
  the whole tuple. It fires whether or not the view was edited. `eproperty.py`'s
  module docstring carries the canonical per-head example and the tuple one.

## How `.output` already works

`Envoy.output` is an identity-preprocess eproperty:

```python
@eproperty
def output(self, value: Any) -> Object:
    return value
```

Its location is `"{self.path}.output"`. The produce side is the controller
installed by `Interleaver.instrument`, which calls
`handle(f"{path}.output", output)` after the module runs. `.input`, `.inputs`, and
`.skip` follow the identical pattern one location over. `SourceEnvoy`'s op-level
`.output`/`.input`/`.inputs` (`src/nnsight/intervention/source.py`) are the same
descriptors, keyed on an operation's path.

## The canonical non-module example: `tracer.result`

The model's own return value is not a module output, yet you can read it. It's an
eproperty on the tracer with no `description` and — because the
tracer has no `path` — a bare `"result"` location:

```python
class InterleavingTracer(Tracer):
    @eproperty
    def result(self, value):
        return value
```

`Envoy.interleave` (`envoy.py`) serves it under that location after the forward:

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
in the repr (`src/nnsight/modeling/vllm/vllm.py`):

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
two sides can't drift out of sync
(`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py`):

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

## Wiring constraint: `envoys=`

An `eproperty` only works on a class that is actually used as an envoy — the base
`Envoy`, an `NNsight`/model subclass, `SourceEnvoy`, the tracer, or a runtime model
like `VLLM`. Child envoys default to the base `Envoy`, so to give a specific
submodule a subclass bearing a custom eproperty, pass `envoys=` — a map from a
module type or dotted path suffix to an `Envoy` subclass (`_resolve_envoy_class`,
`envoy.py`). Without `envoys=`, put the custom eproperty on the model/runtime
subclass (or the tracer).

```python
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

class Heads(Envoy):
    @eproperty(key="output")
    def heads(self, value):                     # preprocess: [B,S,H] -> heads
        b, s, h = value.shape
        return value.view(b, s, self.n_heads, h // self.n_heads).transpose(1, 2)

    @heads.transform
    def heads(self, value, raw):                # write the edited heads back
        b, nh, s, hd = value.shape
        return value.transpose(1, 2).reshape(b, s, nh * hd)

model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    envoys={GPT2MLP: Heads}, dispatch=True,     # or {"mlp": Heads} by path suffix
)
with model.trace(prompt):
    model.transformer.h[0].mlp.heads[:, 5] = 0  # zero head 5; transform swaps it back
```

`Heads` reshapes a bare `[B, S, H]` tensor, so it fits a module whose `.output` is
that tensor (an MLP); on a tuple-output module (attention) index `value[0]` and
return the whole tuple from the transform — see
[per-head-attention.md](../patterns/per-head-attention.md). Non-matching modules stay
the base `Envoy`.

Inside a preprocess, `self._module` is the wrapped `torch.nn.Module`. Read
per-module facts (an attention module's `num_heads`) from it; a module that does not
carry them (a `Conv1D` projection) needs them as a class attribute.

How a key matches:

- A suffix matches the native path or a `rename=` alias (`rename={"attn": "self_attn"}`
  lets `envoys={"self_attn": Heads}` reach GPT-2's `attn`), and aliases compose
  through ancestors: with `rename={"transformer.h": "layers", ...}` the key
  `"layers.0.self_attn"` names what `model.layers[0].self_attn` reaches. One map can
  be written in aliased names and reused across architectures.
- A `*` component matches any one component: `"layers.*"` wraps every block, not
  the container.
- A type or native-path match takes precedence over an alias match.
- An alias from a class-keyed `rename` (`{GPT2Attention: "self_attn"}`) is a
  spelling of that module only: `"self_attn"` matches it, `"self_attn.c_proj"`
  matches nothing. Key the rename by name to reach what is under it.

## Reaching other modules: `parent` and `root`

An eproperty that needs more than its own module's value reaches the rest of the
tree through `self.parent` (the envoy one level up, `None` on the root) and
`self.root` (the model envoy). Reads through them follow the same execution order
as any other read, so a preprocess on attention can read the MLP that runs after it:

```python
class Attn(Envoy):
    @eproperty(key="output")
    def with_mlp(self, value):                  # attention output + the block's MLP output
        return value[0] + self.parent.mlp.output

model = TransformersModel("openai-community/gpt2", envoys={"attn": Attn}, dispatch=True)
with model.trace(prompt):
    summed = model.transformer.h[1].attn.with_mlp.save()
```

The link is a weak reference to the envoy whose `_children` holds this one
(`_parent`). It is set as the child is built (`parent=` on `Envoy.__init__`), kept
through a dispatch re-point (`_update` keeps the envoys), and left out of the
pickled state: on unpickling a parent re-links its children, and an envoy shipped
without its parent has none. Being weak, it keeps no ancestor alive, so `del model`
frees the tree by refcounting. A shared module's envoy has the parent at its first
path; an alias is not a child and changes no parent.

## Shared modules

A module the tree reaches by two paths — a layer a wrapper keeps a second
reference to (vLLM's MLA attention holds `q_proj`/`kv_b_proj`/`o_proj` under
`self_attn` and again under `self_attn.mla_attn`), or tied weights — gets **one**
envoy, at the first path, and the later name is bound as an alias to it, the way
`torch.nn.Module.named_modules()` lists a shared module once. So the module has
one location (`model...self_attn.q_proj.output`), every spelling reaches the same
envoy, and `_aliases` on the aliasing parent records where it points.

The container entry still counts: `layers[2]` and iteration walk every entry the
wrapped `ModuleList` holds — including one whose module an earlier entry already
brought in, and including a container rebuilt from blocks the tree already wraps
(`model.transformer.h = nn.ModuleList(list(model.transformer._module.h)[:4])`).
Each such entry names the one envoy at the module's first path.

## Submodules named after a served value

A submodule may be named after a value nnsight serves: every BERT-style encoder has
an `output` module. The submodule mounts as `E_<name>` and `.output` keeps its usual
meaning, with a warning (`_mount_name`, `envoy.py`). The moved envoy keeps its own
path, so `named_modules()` still reports `...attention.output`, and the repr labels it
`E_output/output`, the same `alias/name` form a `rename` uses. Attribute access is the
only spelling that moved, and `get()` and `rename` go through it, so both are written
against `E_output`. The plain name reaches the served value, which raises outside a
trace, so a `rename` written against it fails at construction. The mount is a plain
attribute rather than a per-envoy subclass overriding the descriptor, which keeps the
envoy picklable by reference for remote runs.

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

- `src/nnsight/intervention/eproperty.py` — the `eproperty` descriptor: the stub
  (preprocess), `postprocess`, `transform`, `__get__`/`__set__`, `provide`
- `src/nnsight/intervention/envoy.py` — the `.input`/`.inputs`/`.output` eproperties;
  `Envoy.interleave` serving `"result"`; `Envoy.__repr__` surfacing described eproperties
- `src/nnsight/intervention/tracer.py` — `result` eproperty
- `src/nnsight/intervention/interleaver.py` — `Mediator.value`/`swap`/`skip`, `Interleaver.handle`, `Interleaver.instrument`
- `src/nnsight/modeling/vllm/vllm.py` — `logits`/`samples` eproperties (a real runtime example)

## Related

- [interleaver-internals.md](./interleaver-internals.md) — `handle` and the event protocol in depth
- [source-internals.md](./source-internals.md) — operation-level locations
- [adding-a-new-runtime.md](./adding-a-new-runtime.md) — where a runtime plugs its values in

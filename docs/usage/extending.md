---
title: Extending nnsight
one_liner: Build custom models by subclassing NNsight/Envoy — batching, attached modules, loaders, custom batchers and tracers.
tags: [usage, extending, envoy, library-development]
related: [docs/usage/access-and-modify.md, docs/usage/invoke-and-batching.md, docs/usage/rename-modules.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/eproperty.py, src/nnsight/modeling/base.py, src/nnsight/intervention/batching.py, src/nnsight/modeling/mixins/loadable.py, src/nnsight/modeling/transformers.py]
---

# Extending nnsight

## What this is for

`NNsight` is a thin, named subclass of `Envoy` — `Envoy` is the node type the model
tree is built from, `NNsight` is the conventional wrapper for a whole model, and
the higher-level classes (`TransformersModel`, `DiffusionModel`, ...) are
specialized envoys that add loading/tokenization on top of the same behavior.

So "extending nnsight" means **subclassing `NNsight`/`Envoy`** and overriding a few
well-defined extension points. Each is an underscore-prefixed attribute or method
with a working default, so you override only the one you need. The surface is:

1. Override `_batch_size` / `_batch` to support batched invokes.
2. Add methods/attributes to a model subclass.
3. Attach standalone modules to the tree so they expose activations.
4. Set `_batcher_class` to a `Batcher` subclass for a non-standard batch layout.
5. Use the `Loadable` mixin to build the model from a repo id / config.
6. Pass a custom `tracer_cls`.
7. Define custom served values with the `eproperty` descriptor.

> `.input` / `.inputs` / `.output` (and `tracer.result`) are `eproperty`
> descriptors, and you can define your own. To attach a custom `eproperty` to a
> **specific** module, pass `envoys=` (a map from a module type or dotted path
> suffix to a custom `Envoy` subclass); otherwise a custom `eproperty` lives on a
> class already used as an envoy (an `NNsight`/model subclass, the tracer, or
> `VLLM`).

## Wrapping any module

```python
import torch
from nnsight import NNsight

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2),
)
model = NNsight(net)                 # root envoy; children auto-wrapped

with model.trace(torch.rand(1, 5)):
    hidden = model[0].output.save()  # index into a Sequential / ModuleList
```

Every submodule is mirrored as an `Envoy` and exposes `.input` / `.output` /
`.source`.

## 1. Batching: `_batch_size` / `_batch`

Base `NNsight` runs a single invoke, but batching two or more raises
`NotImplementedError`. Override two methods to support it:

- `_batch_size(*inputs, **kwargs) -> int` — how many batch rows this invoke's
  input contributes (0 if it's params-only).
- `_batch(invokes, fn) -> (args, kwargs)` — combine the collected invokes
  (`invokes` is a list of `(inputs, kwargs)`) into one call for `fn`.

```python
import torch
import torch.nn as nn
from nnsight import NNsight

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class BatchModel(NNsight):
    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}

model = BatchModel(MLP())
with model.trace() as tracer:
    with tracer.invoke(torch.randn(2, 8)):
        a = model.fc1.output.save()      # this invoke's 2 rows
    with tracer.invoke(torch.randn(3, 8)):
        b = model.fc1.output.save()      # this invoke's 3 rows
```

Row scoping (narrow/widen) is handled by the `Batcher` — you only say how many rows
each input is and how to concatenate them. See
[invoke-and-batching.md](invoke-and-batching.md).

## 2. Adding methods and attributes

A model subclass is a normal Python class — add helpers that run inside a trace:

```python
class MyModel(NNsight):
    def logit_lens(self, hidden):
        return self[1](hidden)           # run a later module ad hoc

model = MyModel(torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 2)))
with model.trace(torch.rand(1, 5)):
    h0   = model[0].output
    lens = model.logit_lens(h0).save()
```

Calling `self[1](hidden)` inside a trace runs that module out of execution
order with the trace stood down, so it serves no values and spends no
occurrences (see `Envoy.__call__`) — the logit-lens idiom.

## 3. Attaching a standalone module

Submodules of the wrapped module are wrapped automatically. To expose a module that
is **not** part of the wrapped module's tree — a streamer, a sampler, a generated-id
passthrough — build an `Envoy` over it with the model's interleaver and add it to
`_children`. This is exactly how `TransformersModel` exposes `model.generator`
(`src/nnsight/modeling/transformers.py`).

For its `.output` to be readable, the standalone module has to actually be *called*
during the run, **with `hook=True`** — that is what leaves the trace watching the
call, so the value reaches a worker parked on `model.extra.output`. A plain
`NNsight` has no `generate` to hang that call on, so give the subclass a driver
method and decorate it with `@traceable`. `with model.run(x):` then traces that
method, and the model's own `__call__` stays untouched:

```python
from nnsight.intervention.envoy import Envoy, traceable

class MyModel(NNsight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra = Envoy(
            ExtraModule(),
            path=f"{self.path}.extra",
            interleaver=self.interleaver,
        )
        self._children.append(self.extra)

    @traceable
    def run(self, x):
        out = self._module(x)
        return self.extra(out, hook=True)     # hook=True, or .output is unreadable

with model.run(torch.ones(1, 5)):
    passed = model.extra.output.save()
```

`@traceable` is sugar for `trace(..., fn=self.run)`; pass `fn=` directly if you'd
rather not decorate. `TransformersModel.generate` is the same shape, with
`self.generator` as the attached child.

Standalone children survive a model-environment rebind (e.g. lazy dispatch swapping
in real weights): they keep their own module and their own place in the tree.

## 4. Custom `Batcher` for non-standard batch layouts

The `Batcher` owns the row math: it slices a full batched activation down to an
invoke's rows on the way to that block (`_narrow_tensor`) and splices an edit back
into the whole batch on the way out (`_widen_tensor`). The base layout is a plain
dim-0 stack. If your model's batch axis isn't that — diffusion's
classifier-free-guidance doubling, vLLM's flat token axis — subclass `Batcher`,
override those two, and name your subclass on the model class:

```python
from nnsight.intervention.batching import Batcher

class MyBatcher(Batcher):
    def _narrow_tensor(self, tensor, group):
        ...                                    # slice to group's rows
    def _widen_tensor(self, full, group, edited):
        ...                                    # splice `edited` back into `full`

class MyModel(NNsight):
    _batcher_class = MyBatcher                 # the whole installation
```

`_batcher_class` is the only wiring needed — both the standard tracer and the vLLM
tracer build the run's batcher as `envoy._batcher_class(envoy, kwargs)`.
`DiffusionModel` is the reference override; `VLLMBatcher`
(`src/nnsight/modeling/vllm/batching.py`) is the worked example of a genuinely
different layout.

## 5. Loading from a repo id / config — the `Loadable` mixin

`Loadable` (`src/nnsight/modeling/mixins/loadable.py`) lets a model be constructed
from something other than a live `nn.Module`: if the first arg is already a module
it's wrapped directly, otherwise `_load(*args, **kwargs)` builds one.

```python
from nnsight.modeling.mixins.loadable import Loadable

class MyLoadable(Loadable):
    def _load(self, repo_id, **kwargs):
        module = build_module_from(repo_id, **kwargs)   # your loader
        return module

model = MyLoadable("my-org/my-model")   # _load runs; result is wrapped
```

`TransformersModel` composes this with a meta-device mixin so the architecture can
be built without weights (for `scan`) and dispatched later.

## 6. Custom tracer

Pass `tracer_cls=` to `trace` to run the block through your own
`InterleavingTracer` subclass — the extension point for a custom `execute` (e.g. a
new runtime, a different batcher, or a fake-tensor mode like `ScanningTracer`):

```python
with model.trace(x, tracer_cls=MyTracer):
    ...
```

## 7. Custom served values: `eproperty`

`.output` / `.input` / `.inputs` and `tracer.result` are `eproperty` descriptors
(`src/nnsight/intervention/eproperty.py`): reading one parks the worker until the
model reaches that location (`"{path}.{key}"`) and returns the value there; writing
one swaps a new value in. Define your own to expose a run-level value the model
produces outside a module's own input/output.

Decorate a stub with `@eproperty` (or `@eproperty(key=..., description=...)`). The
stub **is the preprocess** — it maps the raw served value to what the user reads, so
an identity view is just `return value`. `key` defaults to the method name; several
eproperties can share a key to give different views of one location (that's how
`inputs` shares `input`'s key). Three optional callbacks refine it:

- `.postprocess` — runs on a *written* value before it's swapped in (e.g. repack a
  lone `input` back into the `(args, kwargs)` the model expects).
- `.transform` — the write-back for an edited preprocess *view*: when preprocess
  hands back a reshaped/sliced view, in-place edits to it are invisible to the model
  (which still holds the original), so a transform maps the edited view back to the
  model's layout. It fires once, after the read, and is spliced in like a swap.
- `.provide(obj, value)` — serves the value from the model side (via
  `interleaver.handle`), resuming a worker parked on that location. Call it from your
  runtime where the value is produced.

A `description=` surfaces the eproperty in the Envoy repr tree as
`(name): description`; the plain built-in views carry no description and stay hidden.

`VLLM` (`src/nnsight/modeling/vllm/vllm.py`) defines two this way, served from the
model runner with `type(model).logits.provide(model, ...)`:

```python
from nnsight.intervention.eproperty import eproperty

class VLLM(NNsight):
    @eproperty(description="pre-sampling logits for this step")
    def logits(self, value):        # preprocess: identity view
        return value

    @eproperty(description="token ids drawn from logits this step")
    def samples(self, value):
        return value
```

A preprocess/transform pair gives a reshaped, writable view — e.g. per-head
attention (from the `eproperty` docstring):

```python
class Heads(Envoy):
    @eproperty(key="output")
    def heads(self, value):                     # preprocess: [B,S,H] -> heads
        b, s, h = value.shape
        return value.view(b, s, self.n_heads, h // self.n_heads).transpose(1, 2)

    @heads.transform
    def heads(self, value):                     # write the edited heads back
        b, nh, s, hd = value.shape
        return value.transpose(1, 2).reshape(b, s, nh * hd)

with model.trace(prompt):
    model.attn.heads[:, 5] = 0                  # zero head 5; transform swaps it back
```

To attach this `.heads` accessor to specific modules, pass `envoys=` mapping the
module type (or a dotted path suffix) to the subclass. `Heads` above reshapes a
bare `[B, S, H]` tensor, so it fits a module whose `.output` is that tensor (an
MLP); a module whose `.output` is a tuple (attention) indexes `value[0]` instead:

```python
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    envoys={GPT2MLP: Heads}, dispatch=True,   # or {"mlp": Heads} by path suffix
)
```

Non-matching modules stay the base `Envoy`. See
[per-head-attention.md](../patterns/per-head-attention.md) for the attention
(tuple-output) version.

## Gotchas

- **`envoys=` targets specific modules.** Map a module type or dotted path suffix
  to a custom `Envoy` subclass to attach a custom `eproperty` there; without it a
  custom `eproperty` lives on the model subclass.
- **Batching needs both `_batch_size` and `_batch`,** and the failure lands at
  trace time, not construction time. With only the default, a second input invoke
  raises `NNsight does not support batching multiple invokes`.
- **A batched write has to keep its rows.** Assigning a replacement with a
  different leading dim than the block owns raises
  `A batched write has to keep its rows: ...` before the value reaches the model.
- **A class-level attribute on an `Envoy`/`NNsight` subclass is shared across
  instances** — set per-instance config in `__init__`. (`_batcher_class` is meant
  to be class-level; per-instance config is not.)
- **Attached standalone modules must be called with `hook=True`** to produce a
  readable `.output`. Called without it, the read is never served and the trace
  ends with `'model.extra.output.i0' was requested but the model already ran past
  it` (an `OutOfOrderError`).

## Related

- [access-and-modify.md](access-and-modify.md) — the built-in `.output` / `.input`.
- [invoke-and-batching.md](invoke-and-batching.md) — how `_batch_size` / `_batch`
  feed batching.
- [rename-modules.md](rename-modules.md) — aliasing module paths.

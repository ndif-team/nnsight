---
title: Envoy
one_liner: Envoy wraps a torch.nn.Module and mirrors its submodule tree, exposing .input / .inputs / .output as eproperty descriptors over Mediator.value / Mediator.swap, plus .skip (method) and .source (property).
tags: [concept, mental-model, envoy]
related: [docs/concepts/interleaver-and-controller.md, docs/concepts/source-tracing.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/modeling/base.py]
---

# Envoy

`Envoy`'s served values (`.input`, `.inputs`, `.output`) are `eproperty` descriptors — a small `property` subclass (`intervention/eproperty.py`) that reads/writes an interleaver location; `.skip` is a method and `.source` a plain property.

## What this is for

`Envoy` (`envoy.py`) is the user-facing wrapper around a `torch.nn.Module`. It mirrors the module's submodule tree, so every module is reachable by the same attribute path (`model.transformer.h[0].mlp`), and exposes each module's live values during a forward pass:

- `.output` — the module's forward return value
- `.input` — the first positional argument (or first keyword argument)
- `.inputs` — the full `(args, kwargs)` pair
- `.skip(value)` — bypass this module's forward, using `value` as its output
- `.source` — operation-level access inside the forward (see [Source Tracing](source-tracing.md))

`NNsight(module)` (`modeling/base.py`) is just a root `Envoy` with a conventional name; `TransformersModel`, `DiffusionModel`, etc. are `Envoy` subclasses that add loading/tokenization on top of the same behavior.

## When to use / when not to use

- Wrap any PyTorch model with `nnsight.NNsight(my_module)` and trace it directly.
- Subclass `NNsight`/`Envoy` when you need model-specific loading, input prep, or new served values (see [Extension surface](#extension-surface)).
- Don't replace `Envoy._module`'s `forward` by hand — reassign the module through the envoy so `instrument` re-runs and children are rebuilt.

## Canonical pattern

```python
import torch
from nnsight import NNsight

net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
model = NNsight(net)

with model.trace(torch.ones(1, 4)):
    first = model[0].input.save()      # read input BEFORE output (forward order)
    model[0].skip(model[0].input)      # skip: layer 0 returns its own input
    out   = model[0].output.save()     # so its output is that same input
    model[1].output[:] = 0             # write in place
    final = model.output.save()

assert torch.equal(first, out)
assert torch.equal(final, torch.zeros(1, 2))
```

Every line addresses a location later than the one above it: layer 0's input, its skip gate,
its output, then layer 1. Reorder any two and the run ends in `OutOfOrderError`.

The same shape on a transformer — the skip is armed before the output it decides:

```python
with model.trace("Hello world"):
    model.transformer.h[0].skip(model.transformer.h[0].input)
    hs = model.transformer.h[0].output.save()     # Tensor, shape (1, 2, 768)

assert tuple(hs.shape) == (1, 2, 768)
```

> **Shape note:** in current `transformers`, a GPT-2 block's `.output` is a plain `Tensor` (`(batch, seq, hidden)`), not a `(hidden, ...)` tuple. `print(model.<module>.source)` or a quick `.output.shape.save()` tells you the real shape — don't assume `output[0]` is the hidden state.

## The served values are `eproperty` descriptors

`.input`, `.inputs`, and `.output` are `eproperty` descriptors — a small `property`
subclass (`intervention/eproperty.py`) whose location is `"{self.path}.{key}"`. The
decorated stub is the descriptor's **preprocess**: it takes the raw value the
interleaver served at that location and returns what you read (`envoy.py`–`455`):

```python
@eproperty(key="input")
def inputs(self, value):                 # identity view of the whole (args, kwargs)
    return value

@eproperty
def input(self, value):                  # first-argument view over the same location
    args, kwargs = value
    return first_input(args, kwargs)

@input.postprocess                       # write side: repack the lone arg
def input(self, value):
    args, kwargs = Mediator.value(f"{self.path}.input")
    return replace_first_input(args, kwargs, value)

@eproperty
def output(self, value):                 # identity view of the module's output
    return value
```

- **Reading** the descriptor calls `Mediator.value(location)` — the worker parks
  until the model reaches `location` — then runs the stub (preprocess) on the
  served value and hands you the result.
- **Assigning** runs a registered `.postprocess` (if any) then calls
  `Mediator.swap(location, value)` — the worker parks, and the model side
  substitutes the value in.
- **In-place edits** (`model.layer.output[:] = 0`) work because the identity
  preprocess returns the live tensor and the module's controller returns whatever
  the worker left it as.

`.input` / `.inputs` share the key `"input"` (both address `"{path}.input"`):
`.inputs` returns the whole `(args, kwargs)`; `.input` extracts the first argument
(`first_input`) in its preprocess and repacks it (`replace_first_input`) in its
`postprocess` on write.

A third callback, `.transform`, is the write-back half of a *reshaping* preprocess.
When a preprocess returns a reshaped/sliced view (e.g. a per-head split), in-place
edits to that view are invisible to the model, so a `@output.transform` maps the
edited view back to the model's layout; it fires once, after the block is done with
the read, and its result is spliced in like a swap. The base `.input`/`.output` are
identity views and register none — see the per-head example in `eproperty.py`.

Accessing an eproperty **outside a trace** raises (`Mediator`):

```
ValueError: Cannot access `model.0.output` outside of interleaving
```

## Calling a module inside a trace

`Envoy.__call__(*args, hook=False, **kwargs)` (`envoy.py`) runs a module ad hoc, out of its place in the forward pass — the logit-lens idiom:

```python
with model.trace("The Eiffel Tower is in"):
    hidden = model.transformer.h[5].output           # a Tensor
    logits = model.lm_head(model.transformer.ln_f(hidden))
    tok = logits[:, -1].argmax(-1).save()            # -> [262]  (" the")
```

- **`hook=False` (default) while interleaving:** runs the module the ordinary way, with this trace stood down for the duration. `hook=` names whether *nnsight* watches the call — the module's own PyTorch hooks fire either way, so a runtime that keeps collectives in them (transformers tensor parallelism) still works, and a `forward_hook` you registered yourself fires once for the real forward and once for this call. What `hook=False` buys is that nnsight serves no value and spends no occurrence, for this module *or anything under it*, so the call leaves its real place in the forward untouched.
- **`hook=True`:** leaves the trace armed, so the call goes through the module's controller and its submodules become addressable at `.submodule.output`. Use it for a module *attached* to the tree that isn't part of the real forward — an adapter, LoRA, or SAE applied in an edit.
- **Outside a trace:** always the full `module(...)`.

## Extension surface

By default every child envoy is the base `Envoy`, but a model can pass `envoys=` —
a map from a module type or dotted path suffix to a custom `Envoy` subclass — so a
chosen module gets a subclass that exposes a custom `eproperty` (e.g. a per-head
`.heads` view; see [extending.md](../usage/extending.md) and
[per-head-attention.md](../patterns/per-head-attention.md)). You extend nnsight
several ways:

### 1. Subclass `NNsight` / `Envoy`

For model-specific loading and input handling. Override `_batch_size` (how many batch rows an invoke's input is) and `_batch` (how to combine invokes into one forward) — base `Envoy` supports a single invoke; `TransformersModel` overrides both to batch text/tensors. This is how `TransformersModel`, `DiffusionModel`, and `VLLM` are built.

### 2. Attach modules and watch them with `hook=True`

Add any `nn.Module` as an attribute; it's auto-wrapped as a child envoy. Apply it (with `hook=True` so its internals are observable) inside an **edit** so it runs on every trace:

```python
model.transformer.h[0].adapter = MyAdapter()
with model.edit() as (tracer, edited):
    acts = edited.transformer.h[0].output
    edited.transformer.h[0].output[:] = edited.transformer.h[0].adapter(acts, hook=True)

with edited.trace(prompt):
    inner = edited.transformer.h[0].adapter.inner.output.save()   # now observable
```

### 3. Custom served values via `eproperty`

A new served value is an `eproperty` on the model/runtime class, served from the
driver side with its `.provide`. Because child envoys are always built as the base
`Envoy`, the descriptor goes on the model subclass (or the tracer), not an arbitrary
submodule. This is exactly how the vLLM wrapper adds `.logits` and `.samples`
(`modeling/vllm/vllm.py`):

```python
class VLLM(Remotable):
    @eproperty(description="pre-sampling logits for this step")
    def logits(self, value):            # preprocess: identity view of the served logits
        return value
```

Two things to know before writing one:

- **`key="input"` serves the raw `(args, kwargs)` pair**, not a bare tensor — the same value
  `.inputs` gives you. A preprocess on that key destructures it (`(x,), _ = value`) and a
  transform repacks it (`return ((tensor,), {})`). A preprocess written against a bare tensor
  fails on the pair, and a transform must always return a value of the *location's* shape:
  on a tuple-returning module it returns the whole tuple.
- **A preprocess that raises `AttributeError` is swallowed.** An `eproperty` is a `property`,
  and a `property` getter raising `AttributeError` falls through to `__getattr__`, so the bug
  resurfaces as `'Heads' object (nor its module) has attribute 'heads'` — blaming the
  attribute you asked for rather than the line that failed. If a custom eproperty reports
  itself missing, look inside its preprocess first.

The read side is the descriptor; the produce side calls its `.provide` where the
value is computed — in the vLLM model runner,
`type(model).logits.provide(model, original_logits)`, which forwards to
`model.interleaver.handle("model.logits", ...)` at the eproperty's own location.
`tracer.result` is the same pattern with no `description` and a bare `"result"`
location — `Envoy.interleave` serves it with `handle("result", result)` after the
forward. A `description` is what surfaces an eproperty in the Envoy repr tree as
`(logits): pre-sampling logits for this step`; `.input`/`.output` carry none, so they
stay hidden.

## Shared modules

A module the tree reaches by two paths — a layer a wrapper keeps a second
reference to (vLLM's MLA attention holds `q_proj`/`kv_b_proj`/`o_proj` under
`self_attn` and again under `self_attn.mla_attn`), or tied weights — gets **one**
envoy, at the first path, and the later name is bound as an alias to it, the way
`torch.nn.Module.named_modules()` lists a shared module once. So the module has
one location (`model...self_attn.q_proj.output`), every spelling reaches the same
envoy, and `_aliases` on the aliasing parent records where it points.

## Module renaming (aliases)

`rename={...}` on `NNsight`/`Envoy` binds aliases pointing at the same child envoy (`_bind_aliases`, `envoy.py`). A single-component path (`{"transformer": "gpt"}`) binds wherever it resolves; a multi-component path (`{"transformer.h": "layers"}`) binds on the envoy it resolves from. Aliases are ordinary attributes referencing the same object, so they survive a dispatch re-point with no rebuild.

## Overloaded submodule names

If a submodule's name shadows an `Envoy` attribute (e.g. BERT's `output`), the submodule keeps the name and nnsight's attribute moves to `nns_<name>` (a per-instance subclass), with a warning (`_mount_overloaded`, `envoy.py`).

## Gotchas

- **Read `.input` before `.output` of the same module.** Output runs after input; reversing raises `OutOfOrderError` (see [Threading and Mediators](threading-and-mediators.md)).
- **Property access raises outside a trace.** Wrap in `with model.trace(...)`.
- **`.output` shape is model-specific.** Don't assume a tuple; check with `print(module.source)` or a saved `.shape`.
- **`__call__` defaults to `hook=False` inside a trace** — it stands this trace down for the call, so the module serves nothing and spends no occurrence. Its own PyTorch hooks still fire. Pass `hook=True` to make an attached module's internals observable.
- **`skip(value)` reads before the body runs.** You can read the module's own `.input` and pass it as the replacement (identity skip); the controller offers the input before the skip gate.

## Related

- [Interleaver and Controller](interleaver-and-controller.md) — what `Mediator.value`/`swap` are fulfilled by.
- [Source Tracing](source-tracing.md) — `.source` and the per-operation `SourceEnvoy`.
- [Threading and Mediators](threading-and-mediators.md) — what parks on a property access and what resumes it.
- Source: `src/nnsight/intervention/envoy.py` (`Envoy`), `src/nnsight/modeling/base.py` (`NNsight`), `src/nnsight/intervention/eproperty.py` (the descriptor), `src/nnsight/modeling/vllm/vllm.py` (custom-eproperty example).

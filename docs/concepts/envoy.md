---
title: Envoy
one_liner: Envoy wraps a torch.nn.Module and mirrors its submodule tree, exposing .input / .inputs / .output as eproperty descriptors over Mediator.value / Mediator.swap, plus .skip (method) and .source (property).
tags: [concept, mental-model, envoy]
related: [docs/concepts/interleaver-and-controller.md, docs/concepts/source-tracing.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/aliasing.py, src/nnsight/modeling/base.py]
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
- **Mutate the tree through the envoy, not the wrapped module.** `model.layers = nn.ModuleList(...)` and `model.block.mlp = new_mlp` rebuild the envoys and re-instrument. Assigning on `model._module` instead is not tracked: `model.block.mlp` goes on addressing the module it replaced, which the forward pass no longer runs, so reading it raises `OutOfOrderError`.

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

`.input`, `.inputs` and `.output` are `eproperty` descriptors — a small `property`
subclass (`intervention/eproperty.py`) over the location `"{self.path}.{key}"`:

- **Reading** calls `Mediator.value(location)` — the worker parks until the model
  reaches that location — and hands you the served value.
- **Assigning** calls `Mediator.swap(location, value)`, and the model continues
  with your value.
- **In-place edits** (`model.layer.output[:] = 0`) work because the read returns the
  live tensor and the module's controller returns whatever the worker left it as.

`.inputs` is the whole `(args, kwargs)`; `.input` is a first-argument view over the
same location that repacks on write. Accessing any of them outside a trace raises
``ValueError: Cannot access `model.0.output` outside of interleaving``. How to define
your own (preprocess, `.postprocess`, `.transform`, `.provide`) is in
[extending-envoy.md](../developing/extending-envoy.md).

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

- **Subclass `NNsight` / `Envoy`** for model-specific loading, input prep and
  batching — see [extending.md](../developing/extending.md). This is how
  `TransformersModel`, `DiffusionModel` and `VLLM` are built.
- **Attach a module** as an attribute (it is auto-wrapped as a child envoy) and
  apply it with `hook=True` inside an edit so its internals are observable — see
  [edit.md](../usage/edit.md).
- **Add a served value** with an `eproperty` on the model class, or on an `Envoy`
  subclass that `envoys=` maps a module type or path suffix to (a per-head `.heads`
  view, say) — see [extending-envoy.md](../developing/extending-envoy.md), which also
  covers parent links, shared modules and submodules named `output`.

## Module renaming (aliases)

`rename={...}` on `NNsight`/`Envoy` binds aliases pointing at the same child envoy (`bind`, `aliasing.py`). A single-component path (`{"transformer": "gpt"}`) binds wherever it resolves; a multi-component path (`{"transformer.h": "layers"}`) binds on the envoy it resolves from. A dotted alias binds on the envoy its prefix names (`{"attn.c_proj": "attn.o_proj"}` puts `o_proj` on the attention, not the block). Aliases are ordinary attributes referencing the same object, so they survive a dispatch re-point with no rebuild.

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

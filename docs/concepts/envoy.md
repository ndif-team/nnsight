---
title: Envoy
one_liner: An Envoy wraps a torch.nn.Module and mirrors its submodule tree; inside a trace each one names its module's live values — .input / .inputs / .output — plus .skip() and .source.
tags: [concept, mental-model, envoy]
related: [docs/concepts/interleaver-and-controller.md, docs/concepts/source-tracing.md, docs/usage/access-and-modify.md, docs/developing/extending-envoy.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/modeling/base.py]
---

# Envoy

## What an Envoy is

`Envoy` is the wrapper around a `torch.nn.Module` that you address in a trace. It
mirrors the module's submodule tree, so every module is reachable by its attribute
path (`model.transformer.h[0].mlp`), and exposes that module's live values while the
forward pass runs:

- `.output` — the module's forward return value, exactly as returned (a tensor, or a
  tuple — [usage/access-and-modify.md](../usage/access-and-modify.md#tuple-outputs))
- `.input` — the first positional argument (or first keyword argument)
- `.inputs` — the full `(args, kwargs)` pair
- `.skip(value)` — don't run this module's forward; use `value` as its output
- `.source` — the operations inside the forward ([source-tracing.md](source-tracing.md))

Each property names a location (`{path}.output`, …). **Reading** it parks the worker
until the model gets there and returns the real object; **assigning** it replaces the
value the model continues with; **editing in place** (`output[:] = 0`) works because
the read returned the live tensor. Outside a trace there is nothing to serve, so
access raises ([errors/cannot-access-outside-interleaving.md](../errors/cannot-access-outside-interleaving.md)).

`NNsight(module)` is a root `Envoy`; `TransformersModel`, `DiffusionModel` and `VLLM`
are `Envoy` subclasses that add loading and input preparation.

```python
import torch
from nnsight import NNsight

model = NNsight(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)))

with model.trace(torch.ones(1, 4)):
    first = model[0].input.save()      # input before output (forward order)
    model[0].skip(model[0].input)      # layer 0 returns its own input
    out   = model[0].output.save()     # ... so its output is that input
    model[1].output[:] = 0             # write in place
    final = model.output.save()

assert torch.equal(first, out)
assert torch.equal(final, torch.zeros(1, 2))
```

Every line addresses a location later than the one above it: layer 0's input, its
skip gate, its output, then layer 1. Reorder any two and the run ends in
`OutOfOrderError`.

## Changing the tree

Mutate the tree **through the envoy**: `model.layers = nn.ModuleList(...)` or
`model.block.mlp = new_mlp` rebuilds the envoys and re-instruments. Assigning on
`model._module` instead is not tracked — `model.block.mlp` keeps addressing the
module that no longer runs, and reading it raises `OutOfOrderError`. Aliases for
module names: [usage/rename-modules.md](../usage/rename-modules.md).

## Calling a module inside a trace

Calling an envoy runs its module ad hoc, outside its place in the forward — the
logit-lens idiom:

```python
with model.trace("The Eiffel Tower is in"):
    hidden = model.transformer.h[5].output
    tok = model.lm_head(model.transformer.ln_f(hidden))[:, -1].argmax(-1).save()
```

- **`hook=False` (default)**: nnsight stands the trace down for the call, so it serves
  nothing and spends no occurrence for this module or anything under it; the real
  forward's values are untouched. The module's own PyTorch hooks still fire.
- **`hook=True`**: the call goes through the trace, so its submodules are addressable.
  Use it for a module *attached* to the tree that isn't in the real forward (an
  adapter or SAE applied in an [edit](../usage/edit.md)).

## Extending

Subclass `NNsight` for model-specific loading, input prep and batching
([developing/extending.md](../developing/extending.md)); add a new served value or
per-module-type envoy with `eproperty`
([developing/extending-envoy.md](../developing/extending-envoy.md)).

## Related

- [Interleaver and Controller](interleaver-and-controller.md) — what serves the locations.
- [usage/access-and-modify.md](../usage/access-and-modify.md) — reading and writing values in practice.

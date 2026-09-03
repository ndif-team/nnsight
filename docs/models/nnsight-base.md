---
title: NNsight (Base Wrapper)
one_liner: Wrap any torch.nn.Module to gain trace/intervention access; no tokenizer, no batching.
tags: [models, base]
related: [docs/models/index.md, docs/models/transformers-model.md]
sources: [src/nnsight/modeling/base.py, src/nnsight/intervention/envoy.py, tests/test_modeling.py]
---

# NNsight (Base Wrapper)

## What this is for

`nnsight.NNsight` wraps a `torch.nn.Module` you already have. Constructing one
mirrors every child module as an `Envoy`, so `.trace()`, `.edit()` and
`.session()` work against a model nnsight never loaded. It is the simplest entry
point, and the one with no opinions about input format.

`NNsight` is a thin, named `Envoy` subclass (`base.py`): `Envoy` is the node type
the tree is built from, `NNsight` is the conventional name for wrapping a whole
model, and the higher-level wrappers (`TransformersModel`, `DiffusionModel`, ...)
are specialized envoys that add loading and tokenization on top of the same
behavior.

## When to use it

Use `NNsight` for a custom architecture that is not on HuggingFace, for research
code that builds its model in Python (`torch.nn.Sequential`, hand-built encoders,
GANs, RL policy nets, autoencoders), or whenever you want the intervention
machinery and nothing else.

Reach for a subclass instead when you want HF-style loading from a repo id
([`TransformersModel`](transformers-model.md),
[`DiffusionModel`](diffusion-model.md)), tokenization or `.generate()`
([`TransformersModel`](transformers-model.md)), or batching across several
non-empty `tracer.invoke(...)` calls — base `NNsight` implements neither
`_batch_size()` nor `_batch()`, and a second non-empty invoke raises
`NotImplementedError: NNsight does not support batching multiple invokes`. One
input invoke plus any number of empty invokes does work.

## Loading

```python
import torch
from nnsight import NNsight

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2),
)
model = NNsight(net)     # root envoy; children are auto-wrapped
```

### Constructor

```python
NNsight(module: torch.nn.Module, path="model", interleaver=None, rename=None)
```

| Parameter | Description |
|-----------|-------------|
| `module` | An already-instantiated `torch.nn.Module`. There is no repo loading; the model is wrapped as-is. |
| `path` | Root path name for the Envoy tree (default `"model"`). Rarely set by hand. |
| `interleaver` | Optional `Interleaver` to reuse; a fresh one is created if omitted. |
| `rename` | Optional dict of module-path aliases, e.g. `{"transformer.h": "layers"}`. Both original and aliased paths resolve (`Envoy._bind_aliases`). |

There is **no** `dispatch=`, `device_map=` or `dtype=` here — those belong to the
HF-backed wrappers, along with lazy meta-tensor loading and remote execution. The
module you pass in is the module that is used. Move it with
`module.to("cuda")` before or after wrapping, or with the Envoy's own `.to()` /
`.cuda()` / `.cpu()`, which call through to the module and **return the Envoy**,
so you stay on the wrapper:

```python
model = NNsight(net).to("cuda")     # still an NNsight wrapper
model = model.cpu()                 # still an NNsight wrapper
```

`model.device` gives the first parameter's device and `model.devices` the set of
all of them.

## Canonical pattern

```python
import torch
from nnsight import NNsight

net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))
model = NNsight(net)

with model.trace(torch.randn(1, 8)):
    hidden = model[0].output.save()
    final = model.output.save()

print(tuple(hidden.shape), tuple(final.shape))     # (1, 16) (1, 4)
```

(Verified in `tests/test_modeling.py`.)

### Modifying activations

```python
with model.trace(torch.rand(1, 8)):
    model[0].output[:] = 0                          # in-place
    model[1].output = model[1].output * 2           # replacement
    out = model.output.save()
```

### Empty invokes (batching workaround)

One input invoke plus empty invokes runs the same forward in its own worker,
which covers most of what batching would have been used for:

```python
with model.trace() as tracer:
    with tracer.invoke(torch.rand(1, 8)):
        out_a = model[0].output.save()
    with tracer.invoke():                           # empty invoke = same forward
        out_b = model[1].output.save()
```

For real multi-input batching, subclass `NNsight`/`Envoy` and implement
`_batch_size()` and `_batch()`; `TransformersModel` in
`src/nnsight/modeling/transformers.py` is the reference.

## Special properties

The root wrapper exposes the standard envoy set and nothing more: `model.output`
(the wrapped module's forward output), `model.input` (its first positional arg),
`model.inputs` (the full `(args, kwargs)` pair), and `model._module` (the
underlying `torch.nn.Module`). There is no `tokenizer`, `generator`, `processor`
or `config` — those are added by the HF-backed subclasses, as are `.generate()`,
`.scan()`, `.pipe()` and `.dispatch()`. Asking for one gives
`AttributeError: 'NNsight' object (nor its module) has attribute 'scan'`.

## Gotchas

- **Pass a module, not a repo id.** `NNsight("openai-community/gpt2")` raises
  `AttributeError: 'str' object has no attribute '__dict__'`, which names nothing
  you wrote. Use `TransformersModel("openai-community/gpt2")` for HF repos.
- **Re-wrapping the same module is safe.** Wrapping a module twice re-installs
  its controller rather than stacking (`tests/test_modeling.py` `TestUpdate`,
  `tests/test_multiple_wrappers.py`).
- **Module access order matters.** Inside a single invoke, reading `.output` of a
  later layer before an earlier one can deadlock — see [docs/gotchas/](../gotchas/).
- **`save()` outside a trace raises.** `.save()` / `nnsight.save(...)` errors
  when there is no active trace, and reading `model.output` outside one gives
  `Cannot access 'model.output' outside of interleaving`.

## Related

- [docs/models/transformers-model.md](transformers-model.md) — HF models (adds loading, tokenization, generation)
- [docs/models/index.md](index.md) — full decision tree
- `src/nnsight/intervention/envoy.py` — `Envoy` source
- `src/nnsight/modeling/base.py` — `NNsight` source

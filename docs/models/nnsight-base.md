---
title: NNsight (Base Wrapper)
one_liner: Wrap any torch.nn.Module to gain trace/intervention access; no tokenizer, no batching.
tags: [models, base]
related: [docs/models/index.md, docs/models/transformers-model.md]
sources: [src/nnsight/modeling/base.py:6, src/nnsight/intervention/envoy.py:123, tests/test_modeling.py:857]
---

# NNsight (Base Wrapper)

## What this is for

`nnsight.NNsight` is the root wrapper for any pre-instantiated `torch.nn.Module`. Constructing one recursively mirrors every child module as an `Envoy`, so you can trace, observe, and modify intermediate activations via `.trace()` / `.scan()` / `.edit()` / `.session()`. It is the simplest entry point.

`NNsight` is a thin, named `Envoy` subclass (`base.py:6`) — `Envoy` is the node type the tree is built from, `NNsight` is the conventional name for wrapping a whole model, and the higher-level wrappers (`TransformersModel`, `DiffusionModel`, ...) are specialized envoys that add loading/tokenization on top of the same behavior.

Use it when you already have a `torch.nn.Module` instance and just need NNsight's intervention machinery on top.

## When to use / when not to use

Use `NNsight` when:
- You have a custom architecture not on HuggingFace.
- You're working with research code that builds the model in Python (`torch.nn.Sequential`, hand-built encoders, GANs, RL policy nets, classifiers, autoencoders, etc.).
- You want minimal wrapping with no opinions about input format.

Do not use `NNsight` when:
- You want HF-style loading from a repo id — use [`TransformersModel`](transformers-model.md) or [`DiffusionModel`](diffusion-model.md).
- You want automatic tokenization or `.generate()` — use [`TransformersModel`](transformers-model.md).
- You want input batching across multiple non-empty `tracer.invoke(...)` calls — base `NNsight` does not implement `_batch_size()` / `_batch()`. One input invoke plus any number of empty invokes still works.

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
| `rename` | Optional dict of module-path aliases, e.g. `{"transformer.h": "layers"}`. Both original and aliased paths resolve. See `envoy.py:210` (`_bind_aliases`). |

There is **no** `dispatch=`, **no** `device_map=`, **no** `torch_dtype=` here — those belong to the HF-backed wrappers. Move the model with standard `module.to("cuda")` before or after wrapping, or use the Envoy's own `.to()` / `.cuda()` / `.cpu()`.

### Device movement

`.to()`, `.cuda()`, and `.cpu()` call the underlying module's method but **return the Envoy** (not the raw module), so you stay on the wrapper after moving:

```python
model = NNsight(net).to("cuda")     # still an NNsight wrapper
model = model.cpu()                 # still an NNsight wrapper

with model.trace(torch.rand(1, 5, device="cpu")):
    out = model.output.save()
```

Source: `envoy.py:542` (`.to`), `:557` (`.cpu`), `:564` (`.cuda`). The wrapper also exposes `model.device` (first parameter's device) and `model.devices` (set of all parameter devices).

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

(Verified in `tests/test_modeling.py:857`.)

### Modifying activations

```python
with model.trace(torch.rand(1, 8)):
    model[0].output[:] = 0                          # in-place
    model[1].output = model[1].output * 2           # replacement
    out = model.output.save()
```

### Empty invokes (batching workaround)

Base `NNsight` does not implement batching, so multiple non-empty input invokes raise `NotImplementedError`. One input invoke plus empty invokes still works — an empty invoke runs the same forward in its own worker:

```python
with model.trace() as tracer:
    with tracer.invoke(torch.rand(1, 8)):
        out_a = model[0].output.save()
    with tracer.invoke():                           # empty invoke = same forward
        out_b = model[1].output.save()
```

To support multi-input batching, subclass `NNsight`/`Envoy` and implement `_batch_size()` and `_batch()` (see `TransformersModel` in `src/nnsight/modeling/transformers.py` for a reference).

## Special properties

`NNsight` is an `Envoy`, so the root wrapper exposes only the standard ones:

| Property | Description |
|----------|-------------|
| `model.output` | The wrapped module's forward output |
| `model.input` | First positional arg to the wrapped module |
| `model.inputs` | Full `(args, kwargs)` tuple |
| `model._module` | The underlying `torch.nn.Module` |

There is **no** `tokenizer`, **no** `generator`, **no** `processor`, **no** `config` — those are added by the HF-backed subclasses.

## Limitations

- No tokenization. You pass raw tensors (or whatever your module expects).
- No `.generate()` — subclasses (`TransformersModel`, `DiffusionModel`, `VLLM`) provide their own.
- No multi-input batching (see [Empty invokes](#empty-invokes-batching-workaround)).
- No remote execution by itself. `NNsight` is not a remoteable subclass; `TransformersModel` and `VLLM` are.
- No lazy / meta-tensor loading. The module you pass in is the module that's used.

## Gotchas

- **Pre-loaded module required.** `NNsight("repo/id")` does not work — pass a `torch.nn.Module`. Use `TransformersModel("repo/id")` for HF repos.
- **Re-wrapping the same module is safe.** Wrapping a module twice re-applies hooks rather than stacking them (`tests/test_modeling.py` `TestUpdate` / `test_multiple_wrappers.py`).
- **Module access order matters.** Inside a single invoke, accessing `.output` of a later layer before an earlier one can deadlock — see [docs/gotchas/](../gotchas/).
- **`save()` outside a trace raises.** `.save()` / `nnsight.save(...)` errors when there is no active trace (it was a silent no-op in old nnsight).

## Related

- [docs/models/transformers-model.md](transformers-model.md) — HF models (adds loading, tokenization, generation)
- [docs/models/index.md](index.md) — full decision tree
- `src/nnsight/intervention/envoy.py` — `Envoy` source
- `src/nnsight/modeling/base.py` — `NNsight` source

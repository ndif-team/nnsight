---
title: NNsight (Base Wrapper)
one_liner: Wrap any torch.nn.Module to gain trace/intervention access; no tokenizer, no batching.
tags: [models, base]
related: [docs/models/index.md, docs/models/language-model.md, docs/concepts/envoy-and-eproperty.md]
sources: [src/nnsight/modeling/base.py:8, src/nnsight/intervention/envoy.py, src/nnsight/intervention/batching.py]
---

# NNsight (Base Wrapper)

## What this is for

`nnsight.NNsight` is the root wrapper for any pre-instantiated `torch.nn.Module`. Constructing one recursively wraps every child module in an `Envoy` so you can trace, observe, and modify intermediate activations via `.trace()` / `.scan()` / `.edit()` / `.session()`. It is the simplest entry point — and the base class for `LanguageModel`, `VisionLanguageModel`, `DiffusionModel`, and `VLLM`.

Use it when you already have a `torch.nn.Module` instance and just need NNsight's intervention machinery on top.

## When to use / when not to use

Use `NNsight` when:
- You have a custom architecture not on HuggingFace.
- You're working with research code that builds the model in Python (`torch.nn.Sequential`, hand-built encoders, GANs, RL policy nets, classifiers, autoencoders, etc.).
- You want minimal wrapping with no opinions about input format.

Do not use `NNsight` when:
- You want HF-style loading from a repo ID — use `LanguageModel`, `VisionLanguageModel`, or `DiffusionModel`.
- You want automatic tokenization or `.generate()` — use `LanguageModel`.
- You want input batching across multiple `tracer.invoke(...)` calls with non-empty inputs — base `NNsight` does not implement `_prepare_input()` / `_batch()`. You can still use one input invoke plus any number of empty invokes.

## Loading

```python
from nnsight import NNsight
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2),
)
model = NNsight(net)
```

### Constructor

```python
NNsight(module: torch.nn.Module, *, rename: dict[str, str] | None = None, envoys: ... = None)
```

| Parameter | Description |
|-----------|-------------|
| `module` | An already-instantiated `torch.nn.Module`. There is no repo loading; the model is wrapped as-is. |
| `rename` | Optional dict of module-path aliases (e.g. `{"transformer.h": "layers"}`). See `Envoy` rename docs. |
| `envoys` | Optional override for the descendant Envoy class or a `{module_cls: EnvoyCls}` dict. Subclasses can set this as a class attribute to apply throughout the tree. See `src/nnsight/modeling/base.py:40-65`. |

There is **no** `dispatch=`, **no** `device_map=`, **no** `torch_dtype=` here — those belong to the HF-backed subclasses. Move the model to a device with standard `module.to("cuda")` before or after wrapping; `NNsight` is transparent to device placement.

## Canonical pattern

```python
import torch
from nnsight import NNsight

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2),
)
model = NNsight(net)

with model.trace(torch.rand(1, 5)):
    layer0_out = model[0].output.save()
    final = model.output.save()

print(layer0_out.shape, final.shape)
```

### Modifying activations

```python
with model.trace(torch.rand(1, 5)):
    # in-place
    model[0].output[:] = 0
    # or replacement
    model[1].output = model[1].output * 2
    out = model.output.save()
```

### Empty invokes (batching workaround)

Base `NNsight` does not implement batching, so multiple input invokes will raise `NotImplementedError: Batching is not implemented`. You can still use one input invoke plus empty invokes:

```python
with model.trace() as tracer:
    with tracer.invoke(torch.rand(1, 5)):
        out_a = model[0].output.save()

    with tracer.invoke():       # empty invoke = same forward pass, new thread
        out_b = model[1].output.save()
```

To support multi-input batching, subclass `NNsight` and implement `_prepare_input()` and `_batch()` (see `LanguageModel` in `src/nnsight/modeling/language.py:241` for a reference).

## Special properties

`NNsight` inherits from `Envoy`, so the only special properties on the root wrapper are the standard ones:

| Property | Description |
|----------|-------------|
| `model.output` | The wrapped module's forward output |
| `model.input` | First positional arg to the wrapped module |
| `model.inputs` | Full `(args, kwargs)` tuple |
| `model._module` | The underlying `torch.nn.Module` |
| `model._model` | Legacy alias for `_module` (kept for backwards compat, see `base.py:87`) |

There is **no** `tokenizer`, **no** `generator`, **no** `processor`, **no** `config` — those are added by subclasses.

## Limitations

- No tokenization. You pass raw tensors (or whatever your module expects).
- No `.generate()` — `NNsight` does not define multi-token generation. Subclasses (`LanguageModel`, `DiffusionModel`, `VLLM`) provide their own.
- No multi-input batching. Multiple `tracer.invoke(arg)` calls raise `NotImplementedError` unless you implement `_prepare_input()` / `_batch()`.
- No remote execution by itself. `NNsight` is not a `RemoteableMixin` subclass; `LanguageModel` and `VLLM` are.
- No lazy / meta-tensor loading. The module you pass in is the module that's used; pre-allocate it the way you want.

## Gotchas

- **Pre-loaded module required.** `NNsight(repo_id_string)` does not work — pass a `torch.nn.Module` instance. Use `LanguageModel(repo_id)` for HF repos.
- **Re-wrapping the same module is safe.** `NNsight(my_pytorch_model)` followed by another `NNsight(my_pytorch_model)` properly re-applies hooks rather than stacking them. See [docs/gotchas/](../gotchas/) for details.
- **Module access order matters.** Inside a single invoke, accessing `.output` of layer 5 before layer 2 will deadlock — see [docs/gotchas/order-and-deadlocks.md](../gotchas/order-and-deadlocks.md) if it exists.
- **Subclassing `envoys=`.** If you want all `Linear` layers to use a custom envoy class throughout the tree, set the `envoys` class attribute on your `NNsight` subclass (see `base.py:62`).

## Related

- [docs/models/language-model.md](language-model.md) — for HF causal LMs (extends `NNsight` with loading, tokenization, generation)
- [docs/models/index.md](index.md) — full decision tree
- `src/nnsight/intervention/envoy.py` — Envoy and `eproperty`
- `src/nnsight/modeling/base.py` — `NNsight` source

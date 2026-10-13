---
title: Access and Modify Module Values
one_liner: Use `.output`, `.input`, `.inputs` to read activations; in-place slice or assign to modify.
tags: [usage, intervention]
related: [docs/usage/trace.md, docs/usage/save.md, docs/usage/skip.md, docs/usage/source.md, docs/usage/scan.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/interleaver.py]
---

# Access and Modify Module Values

Every wrapped module (`Envoy`) exposes three properties that read or replace its runtime values during a traced run (`trace`, `generate`, `pipe`, `scan`):

| Property | Returns |
|---|---|
| `module.output` | The module's forward-pass return value |
| `module.input` | The first positional input (or first kwarg value) |
| `module.inputs` | `(args, kwargs)` tuple of all inputs |

Reading one returns the real runtime object: `print`, `.shape`, `.mean()`, `.item()` all work, there is no proxy to unwrap. Outside a traced run they raise `ValueError: Cannot access `<path>` outside of interleaving`; use [scan](scan.md) for shapes without execution.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel
import torch

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello world"):
    model.transformer.wte.output = torch.zeros_like(  # replacement
        model.transformer.wte.output
    )
    model.transformer.h[0].output[:] = 0             # in-place modify

    hidden = model.transformer.h[-1].output.save()   # read

assert tuple(hidden.shape) == (1, 2, 768)
```

**Within one invoke, request modules in the order they execute** (a submodule before the block containing it). The lines above are in that order because the embedding runs before block 0, which runs before block 11. Asking for a later module and then an earlier one raises once the model finishes:

```
nnsight.intervention.interleaver.OutOfOrderError:
'model.transformer.h.0.output.i0' was requested but the model already ran past it
```

See [out-of-order-error](../errors/out-of-order-error.md); to reach modules out of order, use separate invokes ([invoke-and-batching](invoke-and-batching.md)).

## In-place vs replacement

```python
model.transformer.h[0].output[:] = 0               # IN-PLACE: mutates the tensor the model holds
model.transformer.h[0].output = my_new_tensor      # REPLACEMENT: the model continues with a new object
```

In-place mutates storage the forward already holds, so anything aliasing that tensor sees the change; replacement leaves the original alone.

**A replacement with no connection to the value it replaced cuts the autograd graph.** `output = torch.zeros_like(output)` is a tensor autograd has never seen. If that severs every path (a whole block's `.output`), an upstream `.grad` read has nothing to wait for and fails with an object id for a location:

```
OutOfOrderError: '139932764543792.grad.i0' was requested but the model already ran past it
```

If other paths bypass the replaced module (an MLP or attention `.output` — the residual stream survives), upstream `.grad` reads **succeed** and silently return a gradient missing that module's path, even when the forward values are bit-identical. Build the new value out of the old one (`output * 0`, `output + steering`) or write in place (`output[:] = 0`); both keep the graph.

## Tuple outputs

A module's `.output` is whatever it returns. In HuggingFace LLMs the **attention module** returns a tuple; GPT-2 *blocks* return a plain tensor. Check with `isinstance(module.output, tuple)` when unsure. On GPT-2 the attention tuple's second element is `None` unless the model was loaded asking for attention weights.

```python
with model.trace("Hello world"):
    attn_out = model.transformer.h[0].attn.output[0].save()   # first element

    # In-place on the first element
    model.transformer.h[0].attn.output[0][:] = 0

    # Or replace the whole tuple, keeping the other elements
    model.transformer.h[0].attn.output = (
        torch.zeros_like(attn_out),
    ) + model.transformer.h[0].attn.output[1:]
```

`module.output[0] = x` is a `__setitem__` on a tuple and fails. Use the in-place form to modify the values that are there, and the rebuilt tuple when a new tensor (a reshape, a stack) takes the element's place.

## `.input` vs `.inputs`

`.inputs` is the raw `(args, kwargs)` captured before the module runs; `.input` is its first positional argument (or first kwarg value). Setting `.input` repacks into `(args, kwargs)`. `.inputs` is assignable too, and is the only way to edit anything past the first argument:

```python
with model.trace("Hello world"):
    model.transformer.h[1].input = model.transformer.h[1].input * 0
    args, kwargs = model.transformer.h[3].inputs
    model.transformer.h[3].inputs = ((args[0] * 0, *args[1:]), kwargs)
    logits = model.output.logits.save()
```

On the root model, `model.inputs` kwargs carry the tokenizer's `attention_mask` — the way to mask pad positions out of a reduction ([invoke-and-batching](invoke-and-batching.md#reducing-over-real-tokens-the-attention-mask)).

## Cloning before modification

In-place edits happen on the live tensor, so reading `.output` again afterwards returns the modified value. Clone to keep a pre-edit copy:

```python
with model.trace("Hello world"):
    before = model.transformer.h[0].output.clone().save()  # pre-mod
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()           # all zeros
```

## Calling modules directly inside a trace

```python
with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output
    logits = model.lm_head(model.transformer.ln_f(hidden))
    tok = logits[0, -1].argmax(dim=-1).save()
# model.tokenizer.decode(tok) -> ' Paris'
```

Calling an envoy inside a trace runs the module normally but **stands this trace down**: nothing the call touches (the module or its submodules) serves a value or spends an occurrence, so the call above applies `lm_head` out of order without disturbing the trace. The module's own PyTorch hooks still fire — a `forward_hook` you registered on `ln_f` fires twice in the block above.

**`hook=True`** lets the trace watch the call: `module(x, hook=True)` makes the module and its submodules observable. Use it for a module attached to the tree that isn't part of the real forward pass — an SAE, adapter or LoRA applied in an [edit](edit.md). Without it, reading such a module's submodules ends in `OutOfOrderError`. Modules the model calls itself are always watched.

## Overloaded names

Some models have a submodule named `output` or `input` (every BERT-style encoder's attention block has an `output` module). The submodule moves to `.E_output`, with a warning, so `.output` still reads the forward output:

```python
attention = model.encoder.layer[0].attention

with model.trace("Hello World"):
    whole_block = attention.output.save()        # the attention block's output
    inner = attention.E_output.output.save()     # the submodule, and its own output
```

`named_modules()` still lists the submodule as `...attention.output`, but `get(...)` and `rename={...}` use `...attention.E_output`.

## Weight edits are permanent

`.output` is a value handed to you for the run; `.weight` is the module's real `nn.Parameter`, so a write to it mutates the loaded model for every later trace, with no warning:

```python
with model.trace(ids):
    model.transformer.h[5].output[:] = 0           # activation: this run only
with model.trace(ids):
    model.transformer.wte.weight[100] = 0.0        # weight: permanent
```

Save and restore around a weight edit (`saved = w[100].clone()` ... `w.data[100] = saved` in a `finally`), or use [`model.edit()`](edit.md), which `clear_edits()` reverses. Do deliberate weight edits (ROME-style) outside a trace so the permanence is visible at the call site.

Before dispatch, parameters are **meta** tensors — right shape and dtype, no storage. Shape-preserving arithmetic on them succeeds and fails later, in torch's words rather than nnsight's. Pass `dispatch=True` (or run a trace) before reading weights.

## Gotchas

- A tensor you build inside the block (`torch.randn(768)`) is on CPU; only the inputs passed to `trace` are moved for you. Derive it from the activation it combines with — `torch.randn(hs.shape[-1], device=hs.device, dtype=hs.dtype)` — or you get `Expected all tensors to be on the same device`. Under `device_map="auto"` each layer can be on a different device.
- Bypassing a module's compute: [skip](skip.md). Operations inside a module's forward: [source](source.md).

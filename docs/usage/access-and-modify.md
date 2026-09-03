---
title: Access and Modify Module Values
one_liner: Use `.output`, `.input`, `.inputs` to read activations; in-place slice or assign to modify.
tags: [usage, intervention]
related: [docs/usage/trace.md, docs/usage/save.md, docs/usage/skip.md, docs/usage/source.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/interleaver.py]
---

# Access and Modify Module Values

## What this is for

Every wrapped module (`Envoy`) exposes three properties that read or replace the module's runtime values during the forward pass:

| Property | Returns |
|---|---|
| `module.output` | The module's forward-pass return value |
| `module.input` | The first positional input (or first kwarg value) |
| `module.inputs` | `(args, kwargs)` tuple of all inputs |

These are ordinary Python properties on `Envoy` (`src/nnsight/intervention/envoy.py`). Reading one calls `Mediator.value(...)`, which blocks the trace's greenlet worker until the model's forward pass produces that value; assigning to one calls `Mediator.swap(...)`, which hands the model a new value to continue with.

## When to use / when not to use

- Use inside a `with model.trace(...)` / `model.generate(...)` / `model.pipe(...)` / `model.scan(...)` body to read activations.
- Use slice assignment `[:] = value` for in-place modification.
- Use direct assignment `module.output = value` for replacement.
- Outside a tracing context, accessing `.output` raises `ValueError: Cannot access `<path>` outside of interleaving`.

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

The three lines are in that order because the embedding runs before block 0, which runs
before block 11. Reading the last block first and then writing an earlier one raises
`OutOfOrderError` — see [Forward-pass-order rule](#forward-pass-order-rule).

## In-place vs replacement

```python
# IN-PLACE: mutates the tensor the model is holding.
# All later reads through the same location see the mutation.
model.transformer.h[0].output[:] = 0

# REPLACEMENT: hands the model a new value (Mediator.swap);
# downstream computation continues with my_new_tensor.
model.transformer.h[0].output = my_new_tensor
```

Both take effect and they are not interchangeable. In-place mutates storage the forward pass
already holds, so anything else aliasing that tensor sees the change; replacement hands the
model a different object and leaves the original alone.

One consequence catches people writing gradients: **a replacement that has no connection to
the value it replaced cuts the graph.** `output = torch.zeros_like(output)` produces a tensor
autograd has never seen, so a `.grad` read on anything captured upstream has nothing to wait
for and fails with an object id for a location:

```
OutOfOrderError: '139932764543792.grad.i0' was requested but the model already ran past it
```

`output[:] = 0` writes through the existing tensor and keeps the graph (the gradient is then
zero, which is usually what you wanted). So is any replacement derived from the old value —
`output = output * 0`, `output = output + steering`. Build the new value out of the old one,
or write in place.

## Tuple outputs

Some modules return a tuple. In the HuggingFace LLMs, the **attention module** returns `(attn_out, ...)`. (In current `transformers`, GPT-2 transformer *blocks* return a plain tensor — `model.transformer.h[0].output` is a tensor, not a tuple — but attention still returns a tuple. Check with `isinstance(module.output, tuple)` when unsure.)

On GPT-2 that tuple is length 2 and its second element is `None` unless the model was loaded
asking for attention weights — so it is a slot, not a value you can read. The recipes below
carry it along without looking at it, which is what you want either way.

```python
with model.trace("Hello world"):
    full = model.transformer.h[0].attn.output   # tuple
    attn_out = full[0].save()                    # first element (tensor)

    # In-place on the first element
    model.transformer.h[0].attn.output[0][:] = 0

    # Replace the whole tuple (keep the other elements)
    model.transformer.h[0].attn.output = (
        torch.zeros_like(attn_out),
    ) + model.transformer.h[0].attn.output[1:]
```

## `.input` vs `.inputs`

`.inputs` returns the raw `(args, kwargs)` tuple captured before the module runs. `.input` is a convenience returning the first positional argument (or the first kwarg value if there are none) — built on top of `.inputs`. Setting `.input` correctly repacks into `(args, kwargs)`:

```python
with model.trace("Hello world"):
    args, kwargs = model.transformer.h[0].inputs
    first = model.transformer.h[0].input.save()

    # Set: repacks into the full (args, kwargs) for the model
    model.transformer.h[1].input = model.transformer.h[1].input * 0
```

`.inputs` is assignable too, and it is the only way to edit anything past the first argument
— you hand back the whole pair:

```python
with model.trace("Hello world"):
    args, kwargs = model.transformer.h[3].inputs
    model.transformer.h[3].inputs = ((args[0] * 0, *args[1:]), kwargs)
    logits = model.output.logits.save()
```

That is exactly what `.input = value` does for you when the value is the first argument.

## Cloning before modification

In-place modifications happen on the live tensor — reading `.output` again after a modification returns the modified value. Clone first to keep a pre-mod copy:

```python
with model.trace("Hello world"):
    before = model.transformer.h[0].output.clone().save()  # pre-mod
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()           # post-mod
# before is not all-zeros; after is all-zeros
```

The `.clone()` is a real `torch.Tensor.clone()` — the worker receives the actual tensor.

## Forward-pass-order rule

Within a single invoke, request modules in the order they execute. Asking for a later module and then an earlier one deadlocks and raises `OutOfOrderError` once the model finishes:

```
nnsight.intervention.interleaver.OutOfOrderError:
'model.transformer.h.0.output.i0' was requested but the model already ran past it
```

To access modules out of order, use separate invokes — see `docs/usage/invoke-and-batching.md`.

## Calling modules directly inside a trace

```python
with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output
    # Calling the envoy runs the module with this trace stood down, so it
    # applies lm_head out of order WITHOUT re-triggering interleaving.
    logits = model.lm_head(model.transformer.ln_f(hidden))
    tok = logits[0, -1].argmax(dim=-1).save()
# model.tokenizer.decode(tok) -> ' Paris'
```

While interleaving, `Envoy.__call__` runs the module normally but stands *this trace* down, so nothing the call touches — the module or its submodules — serves a value or spends an occurrence. The module's own PyTorch hooks still fire: `hook=` names whether nnsight watches the call, not whether the module runs its hooks. A `forward_hook` you registered on `ln_f` yourself fires twice in the block above — once in the real forward, once for the ad-hoc call. Pass `hook=True` to let the trace watch it too (its submodules become observable) — used for a module attached to the tree that isn't part of the real forward pass (an adapter/LoRA/SAE applied in an edit). See `Envoy.__call__`.

## Overloaded names

If a module's class has a submodule named `input`, `output`, `inputs`, etc. (e.g. BERT's `output`), the submodule keeps that name and nnsight's property moves to `.nns_output` (with a warning). See `Envoy._mount_overloaded`.

## Module skipping / source access

- `module.skip(replacement)` bypasses a module's compute — see `docs/usage/skip.md`.
- `module.source.<op>.output` reaches operations inside a module's forward — see `docs/usage/source.md`.

## Gotchas

- Within an invoke, access modules in forward-pass order or hit `OutOfOrderError`. See `docs/errors/out-of-order-error.md`.
- For tuple-returning modules, `module.output[0] = x` is a `__setitem__` on a tuple and fails. Use `module.output[0][:] = x` (in-place on the first element) or rebuild the tuple and assign to `module.output`.
- Reading `.output` returns the real runtime tensor — `print`, `.shape`, `.mean()`, `.item()` all work; there is no proxy to unwrap.
- Replacing an activation with a tensor built from scratch (`torch.zeros_like(...)`) severs autograd. Derive the replacement from the old value, or write in place.
- Outside interleaving, `.output` raises `ValueError: Cannot access ... outside of interleaving`. Use `model.scan(...)` for shapes without execution.

## Related

- `docs/usage/trace.md`
- `docs/usage/save.md`
- `docs/usage/skip.md`
- `docs/usage/source.md`
- `docs/usage/scan.md`

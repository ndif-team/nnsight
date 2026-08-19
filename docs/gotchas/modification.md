---
title: Modification Pitfalls
one_liner: Common mistakes when modifying activations — in-place vs replacement, tuple outputs, and aliasing the "before" state.
tags: [gotcha, intervention, modify]
related: [docs/usage/access-and-modify.md, docs/gotchas/cross-invoke.md]
sources: [src/nnsight/intervention/envoy.py:448, src/nnsight/intervention/eproperty.py, src/nnsight/intervention/interleaver.py:279]
---

# Modification Pitfalls

## TL;DR
- `output[:] = v` mutates the existing tensor in place; `output = v` rebinds the name **and** schedules a `SWAP` event that replaces what the model sees downstream. They are not interchangeable, but both take effect.
- A GPT-2 (transformers 5+) **block** `.output` is a **plain tensor** `(batch, seq, hidden)` — index/edit it directly. Do **not** write `output[0]` expecting a tuple. Some *sub*modules (e.g. attention) still return tuples — for those, `.output[0]` is the tensor.
- Assigning into a tuple output — `attn.output[0] = t` — raises `TypeError` (tuples don't support item assignment). Rebuild the tuple and assign the whole thing, or edit in place: `attn.output[0][:] = ...`.
- To keep the "before" state of a value you're about to mutate, `.clone().save()` it first — otherwise `before` and `after` alias the same modified tensor.
- Prefer replacing a **whole** tensor over an in-place slice-assign into a *tuple element* view across a barrier — the latter can crash. Assign the whole value instead.
- Editing an **activation** is scoped to that run; editing a **weight** is permanent, and the syntax looks identical. There is no warning.

---

## In-place `[:] = ` vs replacement `=`

### Symptom
`output[:] = 0` visibly changes the model; `output = torch.zeros(...)` also changes the model but via a different mechanism. Confusion when one is expected and the other used.

### Cause
- `output[:] = v` is `__setitem__` on the tensor `.output` handed you. It mutates storage the forward pass already holds a reference to, so the change is visible.
- `output = v` is a Python rebind that also fires the eproperty **setter**: `.output` is an `eproperty` (`src/nnsight/intervention/eproperty.py`) whose `__set__` calls `Mediator.swap(...)`, sending a `SWAP` event so the interleaver substitutes your value into the forward pass for the rest of the run.

Both work; they differ in what they touch. In-place edits the existing tensor (other references see it); replacement substitutes a new tensor downstream (the original object is untouched).

### Wrong code
```python
with model.trace("Hello"):
    # attention still returns a tuple — item assignment on a tuple fails
    # TypeError: 'tuple' object does not support item assignment
    model.transformer.h[0].attn.output[0] = torch.zeros_like(
        model.transformer.h[0].attn.output[0]
    )
```

### Right code
```python
with model.trace("Hello world"):
    # a block's output is a plain tensor — edit it directly
    model.transformer.h[0].output[:] = 0

    # or replace the whole tensor (the setter schedules a SWAP)
    model.transformer.h[0].output = torch.zeros_like(model.transformer.h[0].output)
```

For a tuple-returning submodule (attention), edit in place or rebuild the tuple:
```python
with model.trace("Hello world"):
    model.transformer.h[0].attn.output[0][:] = 0          # in-place on the tensor
    # OR replace the whole tuple:
    out = model.transformer.h[0].attn.output
    model.transformer.h[0].attn.output = (torch.zeros_like(out[0]),) + tuple(out[1:])
```

### Mitigation / how to spot it early
- Ask "am I mutating storage, or substituting a new value?" Both are valid; just never write `output[0] = new_tensor` on a tuple.
- `print(module.output)` inside the trace shows whether you have a tensor or a tuple.

---

## Tensor vs tuple outputs

### Symptom
`AttributeError: 'tuple' object has no attribute 'shape'`, or `TypeError: 'tuple' object does not support item assignment`.

### Cause
In transformers 5+, transformer **blocks** return a plain tensor, so `model.transformer.h[i].output` *is* the hidden state `(batch, seq, hidden)`. But some submodules still return tuples — the **attention** module returns `(attn_out, attn_weights)`, so `.output` is that tuple and tensor ops live on `.output[0]`.

Verified structure on GPT-2:
```python
with model.trace("Hello world"):
    print(type(model.transformer.h[0].output).__name__)        # Tensor, shape (1, 2, 768)
    print(type(model.transformer.h[0].attn.output).__name__)   # tuple, len 2
```

### Mitigation / how to spot it early
- Don't assume. `print(module.output)` inside a trace, or `print(module.source)`, reveals the return structure.
- A one-step `model.scan(...)` surfaces the shape/tuple structure without running the model.

---

## Saving the "before" state of an in-place edit

### Symptom
You save `before`, then mutate in place, then save `after` — both come out identical (the modified value).

### Cause
`.save()` records the object, not a snapshot. If `before` aliases the tensor you mutate, the in-place edit is visible through both. `.clone()` first so `before` points at independent storage.

### Right code
```python
with model.trace("Hello world"):
    before = model.transformer.h[0].output.clone().save()
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()
# before holds the original, after holds the zeros
```

---

## Activation patching across invokes needs `.clone()`

### Symptom
A slice captured in invoke 1 and written in invoke 2 behaves like it was overwritten, or `RuntimeError: ... modified by an inplace operation`.

### Cause
`clean_hs = module.output[:, -1, :]` is a *view* into the batched activation. When invoke 2 writes back into the same batch rows, the read-then-write can collapse. `.clone()` materializes an independent tensor.

### Right code
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[:, -1, :].clone()   # independent
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[:, -1, :] = clean_hs
        patched = model.output.logits.save()
```

(See [cross-invoke.md](cross-invoke.md) for why the barrier is needed here.)

---

## A tuple `.output` needs its elements edited, not reassigned

### Symptom
`TypeError: 'tuple' object does not support item assignment` from
`h[5].attn.output[0] = new_attn`.

### Cause
Some modules return a tuple (an attention block's `(output, weights)`), and
Python tuples are immutable. The *tensors inside* the tuple are not — and
`.output` hands back the live ones.

### Two ways to write
```python
out = model.transformer.h[5].attn.output

# Edit the existing tensor in place — writes straight through, no assignment.
out[0][:, -1, :] = clean

# Put a *different* tensor in its place — rebuild the tuple and assign that.
model.transformer.h[5].attn.output = (new_attn,) + tuple(out[1:])
```

Use the first when you are modifying the values that are there; use the second
when the replacement is a new tensor (a reshape, a stack, an arithmetic result)
that has to take the element's place.

---

## A weight edit inside a trace is permanent

### Symptom
Two edits that read almost identically behave completely differently across runs:

```python
with model.trace(ids):                       # ACTIVATION -- scoped to this run
    model.transformer.h[5].output[:] = 0

with model.trace(ids):                       # WEIGHT -- permanent
    model.transformer.wte.weight[100] = 0.0
```

```
activation write: run differs from base: True | NEXT run back to baseline: True
weight write    : run differs from base: True | NEXT run back to baseline: False
                                              | wte[100] still zero: True
```

Every later trace in the process — and anything else holding that model — now runs
against a modified checkpoint.

### Cause
Not an nnsight behaviour: `.output` is a value the interleaver hands you for the
duration of the run, while `.weight` is the module's real `nn.Parameter`. Writing
to it is an ordinary in-place mutation of the loaded model, and the trace block
does not scope it. The trap is that `with model.trace(...)` reads like a scope for
everything inside it.

### Fix
Save and restore around the edit, or work on a copy:

```python
saved = model.transformer.wte.weight[100].clone()
try:
    with model.trace(ids):
        model.transformer.wte.weight[100] = 0.0
        out = model.lm_head.output.save()
finally:
    model.transformer.wte.weight.data[100] = saved
```

For a persistent-but-reversible change, prefer [`model.edit()`](../usage/edit.md),
which stores the intervention and can be undone with `clear_edits()`. For a
genuine weight edit (ROME-style), do it deliberately and outside a trace, so the
permanence is visible at the call site.

Reading weights is of course fine, and composes with activations inside a trace.
One related trap: before dispatch, a model's parameters are **meta** tensors —
correct shape and dtype, no storage. Shape-preserving arithmetic on them succeeds
silently and fails somewhere later, in torch's words rather than nnsight's. Pass
`dispatch=True` (or run a trace) before reading weights.

## Related
- [docs/usage/access-and-modify.md](../usage/access-and-modify.md) — reading and writing module values.
- [docs/gotchas/cross-invoke.md](cross-invoke.md) — barrier rules for cross-invoke patches.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — module-access order constraints.

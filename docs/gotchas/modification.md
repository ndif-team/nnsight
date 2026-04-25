---
title: Modification Pitfalls
one_liner: Common mistakes when modifying activations — in-place vs replacement, tuple outputs, and aliasing the "before" state.
tags: [gotcha, intervention, modify]
related: [docs/usage/access-and-modify.md, docs/gotchas/cross-invoke.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/interleaver.py:198]
---

# Modification Pitfalls

## TL;DR
- `output[:] = v` mutates the existing tensor in place; `output = v` rebinds (and triggers a `swap` event that replaces what the model sees). They are not interchangeable.
- Most transformer blocks return tuples — index `output[0]` to reach the hidden states; assigning to `output[0]` directly does not work, you must do in-place `output[0][:] = ...` or replace the whole tuple `output = (new, *output[1:])`.
- If you want to keep the "before" state of a value you're about to mutate, `.clone().save()` it first — otherwise `before` and `after` alias the same modified tensor.
- For activation patching where two invokes both read the same module, `.clone()` the captured slice or it gets overwritten when the second invoke writes.

---

## In-place `[:] = ` vs replacement `=`

### Symptom
You expect a fresh tensor in the model's forward path, but downstream computations behave as if the original was used (or vice versa). Or: `output[0][:] = 0` works but `output[0] = torch.zeros(...)` silently does nothing visible to the model.

### Cause
Two completely different mechanisms:

- `output[:] = v` is a Python `__setitem__` on the tensor returned by `.output`. It mutates the underlying storage. The model's forward pass already holds a reference to that tensor, so the mutation is visible.
- `output = v` is a Python rebind on a name in *your* worker thread. Without nnsight, that would just shadow the local name and the model would never know. nnsight intercepts this via the `eproperty.__set__` descriptor (`src/nnsight/intervention/interleaver.py:306`), which sends a `SWAP` event to the mediator so the batcher actually replaces the value the model uses for the rest of the forward pass.

So both work, but they have different semantics:

- In-place edits the existing tensor; references to it elsewhere see the change.
- Replacement substitutes a *new* tensor for downstream code; the original tensor is unchanged.

The two get conflated when users try `output[0] = new_tensor` on a tuple output — that's a `__setitem__` on a *tuple*, which raises `TypeError`.

### Wrong code
```python
with model.trace("Hello"):
    # tuple output — TypeError: 'tuple' object does not support item assignment
    model.transformer.h[0].output[0] = torch.zeros(1, 5, 768)
```

### Right code
```python
with model.trace("Hello"):
    # in-place on the first tuple element (mutates the existing hidden state)
    model.transformer.h[0].output[0][:] = 0

    # OR replace the whole tuple (the eproperty __set__ schedules a swap)
    h0 = model.transformer.h[0].output
    model.transformer.h[0].output = (torch.zeros_like(h0[0]),) + h0[1:]
```

### Mitigation / how to spot it early
- Ask "am I mutating storage, or substituting a new value?" If you want either, both are valid; just don't write `output[0] = new_tensor` on a tuple.
- `model.scan(input)` will surface the tuple-vs-tensor structure so you can choose the right pattern before running.

---

## Tuple outputs

### Symptom
`AttributeError: 'tuple' object has no attribute 'shape'`, or `TypeError: 'tuple' object does not support item assignment`. Confusion about why `model.transformer.h[0].output` doesn't behave like a tensor.

### Cause
Most HuggingFace transformer blocks return a tuple `(hidden_states, attention_weights, ...)` (or `(hidden_states, present_key_value)` etc.) — `.output` faithfully gives you that tuple. Tensor operations and `.shape` are on the *first* element, not the tuple itself.

### Wrong code
```python
with model.trace("Hello"):
    # tuple has no .shape
    print(model.transformer.h[0].output.shape)

    # tuple does not support __setitem__
    model.transformer.h[0].output[0] = my_replacement
```

### Right code
```python
with model.trace("Hello"):
    # access the first element
    hs = model.transformer.h[0].output[0]
    print(hs.shape)

    # in-place modification of the first element
    model.transformer.h[0].output[0][:] = 0

    # full-tuple replacement when you need a different first element
    out = model.transformer.h[0].output
    model.transformer.h[0].output = (my_replacement,) + out[1:]
```

### Mitigation / how to spot it early
- `print(model.transformer.h[0].output)` inside the trace prints the actual tuple — useful for confirming structure.
- `print(model)` shows the module type but not the return shape; running a one-step `model.scan(...)` is the quickest way to see the tuple structure.

---

## Saving the "before" state of an in-place edit

### Symptom
You save `before` and then save `after` with an in-place modification between them — both come out identical (and equal to the modified value).

### Cause
`.save()` records the *id* of the object, not a snapshot. If `before` aliases the same tensor that `after` does, an in-place edit is visible through both. The fix is to `.clone()` before the modification so `before` points at a separate tensor whose storage is unaffected.

### Wrong code
```python
with model.trace("Hello"):
    before = model.transformer.h[0].output[0].save()   # alias
    model.transformer.h[0].output[0][:] = 0
    after = model.transformer.h[0].output[0].save()
# before and after both contain the zeroed tensor
```

### Right code
```python
with model.trace("Hello"):
    before = model.transformer.h[0].output[0].clone().save()
    model.transformer.h[0].output[0][:] = 0
    after = model.transformer.h[0].output[0].save()
# before holds the original, after holds the zeros
```

### Mitigation / how to spot it early
- Any time you save a tensor and *also* mutate the same activation, clone the saved one.
- Replacement (`output = new`) doesn't have this problem because the new tensor and the original are already different objects — but the original isn't going to be visible to downstream operations either.

---

## Activation patching needs `.clone()` for cross-invoke same-module patches

### Symptom
You capture a slice in invoke 1 and use it in invoke 2, but the patched value behaves like it was overwritten or has unexpected content. Sometimes errors like `RuntimeError: ... has been modified by an inplace operation`.

### Cause
A capture like `clean_hs = model.transformer.h[5].output[0][:, -1, :]` is a *view*, not a copy. When invoke 2 then writes `model.transformer.h[5].output[0][:, -1, :] = clean_hs`, the assignment uses the (now overwritten) underlying storage in the second invoke's batched activation slot. Because of how the batcher handles slicing across the combined batch, the read-then-write can collapse into a no-op or corrupt the slice.

`.clone()` materializes the view as an independent tensor, so the value captured in invoke 1 is not affected by invoke 2's writes.

### Wrong code
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :]   # view
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[0][:, -1, :] = clean_hs   # may not behave
        patched = model.lm_head.output.save()
```

### Right code
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :].clone()   # independent
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[0][:, -1, :] = clean_hs
        patched = model.lm_head.output.save()
```

### Mitigation / how to spot it early
- Whenever you slice into `.output` and pass the slice to another invoke, `.clone()` it.
- This is the same root cause as the "save aliases the modified tensor" gotcha above — views and in-place writes don't mix.

---

## Related
- [docs/usage/access-and-modify.md](../usage/access-and-modify.md) — full reference for reading and writing module outputs.
- [docs/gotchas/cross-invoke.md](cross-invoke.md) — barrier rules for cross-invoke patches.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — module-access order constraints (related to write semantics).

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
- In transformers <5, transformer blocks returned a tuple `(hidden, ...)` — indexing `output[0]` was required. In transformers 5+ they return a plain tensor, so `output[:] = ...` and `output = ...` work directly. Some submodules (e.g. attention) still return tuples; index `output[0]` for those.
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

The two get conflated when users try `output[0] = new_tensor` on a tuple output (e.g. attention modules, which still return tuples) — that's a `__setitem__` on a *tuple*, which raises `TypeError`.

### Wrong code
```python
with model.trace("Hello"):
    # attention modules still return a tuple in transformers 5+ —
    # TypeError: 'tuple' object does not support item assignment
    model.transformer.h[0].attn.output[0] = torch.zeros_like(model.transformer.h[0].attn.output[0])
```

### Right code
```python
with model.trace("Hello"):
    # transformer blocks return a tensor in transformers 5+ — modify directly
    model.transformer.h[0].output[:] = 0

    # OR replace the whole tensor (the eproperty __set__ schedules a swap)
    model.transformer.h[0].output = torch.zeros_like(model.transformer.h[0].output)

    # For modules that DO still return a tuple (e.g. attention), use in-place
    # on the first element or rebuild the tuple:
    model.transformer.h[0].attn.output[0][:] = 0
    attn_out = model.transformer.h[0].attn.output
    model.transformer.h[0].attn.output = (torch.zeros_like(attn_out[0]),) + attn_out[1:]
```

### Mitigation / how to spot it early
- Ask "am I mutating storage, or substituting a new value?" Both are valid; just don't write `output[0] = new_tensor` on a tuple-returning module.
- `model.scan(input)` will surface the tuple-vs-tensor structure so you can choose the right pattern before running.

---

## Tuple outputs

### Symptom
`AttributeError: 'tuple' object has no attribute 'shape'`, or `TypeError: 'tuple' object does not support item assignment`. Confusion about why `module.output` doesn't behave like a tensor.

### Cause
Some submodules return a tuple instead of a tensor. The most common in HuggingFace models is the **attention module**, which returns `(attn_out, attn_weights)`. `.output` faithfully gives you that tuple; tensor operations and `.shape` are on the *first* element, not the tuple itself.

In transformers <5, transformer blocks themselves also returned tuples. As of transformers 5+, the blocks return a plain tensor, so `model.transformer.h[i].output` *is* the hidden state directly — but submodules like `attn` still return tuples.

### Wrong code
```python
with model.trace("Hello"):
    # attention output is a tuple — has no .shape
    print(model.transformer.h[0].attn.output.shape)

    # tuple does not support __setitem__
    model.transformer.h[0].attn.output[0] = my_replacement
```

### Right code
```python
with model.trace("Hello"):
    # access the first element of the attention tuple
    attn_out = model.transformer.h[0].attn.output[0]
    print(attn_out.shape)

    # in-place modification of the first element
    model.transformer.h[0].attn.output[0][:] = 0

    # full-tuple replacement when you need a different first element
    out = model.transformer.h[0].attn.output
    model.transformer.h[0].attn.output = (my_replacement,) + out[1:]
```

### Mitigation / how to spot it early
- `print(module.output)` inside the trace prints the actual value — useful for confirming whether you have a tensor or a tuple.
- `print(model)` shows the module type but not the return shape; running a one-step `model.scan(...)` is the quickest way to see the structure.

---

## Saving the "before" state of an in-place edit

### Symptom
You save `before` and then save `after` with an in-place modification between them — both come out identical (and equal to the modified value).

### Cause
`.save()` records the *id* of the object, not a snapshot. If `before` aliases the same tensor that `after` does, an in-place edit is visible through both. The fix is to `.clone()` before the modification so `before` points at a separate tensor whose storage is unaffected.

### Wrong code
```python
with model.trace("Hello"):
    before = model.transformer.h[0].output.save()   # alias
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()
# before and after both contain the zeroed tensor
```

### Right code
```python
with model.trace("Hello"):
    before = model.transformer.h[0].output.clone().save()
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()
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
A capture like `clean_hs = model.transformer.h[5].output[:, -1, :]` is a *view*, not a copy. When invoke 2 then writes `model.transformer.h[5].output[:, -1, :] = clean_hs`, the assignment uses the (now overwritten) underlying storage in the second invoke's batched activation slot. Because of how the batcher handles slicing across the combined batch, the read-then-write can collapse into a no-op or corrupt the slice.

`.clone()` materializes the view as an independent tensor, so the value captured in invoke 1 is not affected by invoke 2's writes.

### Wrong code
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[:, -1, :]   # view
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[:, -1, :] = clean_hs   # may not behave
        patched = model.lm_head.output.save()
```

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

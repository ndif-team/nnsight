---
title: Scan
one_liner: Validate shapes/operations under FakeTensor mode without running the real model (`model.scan(...)`).
tags: [usage, scan, validation]
related: [docs/usage/trace.md, docs/usage/save.md, docs/usage/access-and-modify.md, docs/usage/cache.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/modeling/mixins/meta.py]
---

# Scan

`model.scan(input)` runs the traced forward under PyTorch's `FakeTensorMode`: tensors carry shape and dtype but no data, no kernels run, and **the model is not dispatched**, so it works on a meta-loaded model without loading weights. Use it to read activation shapes, or to catch index-out-of-range and shape-mismatch errors early. Everything a trace offers (`.output`, `.input`, `tracer.invoke`, `tracer.cache`, ...) works, and values still need [`.save()`](save.md) to leave the block — `nnsight.save(...)` for ints and lists. Read [What a scan cannot tell you](#what-a-scan-cannot-tell-you) before treating a passing scan as a green light.

## Canonical pattern

```python
import nnsight
from nnsight.modeling.transformers import TransformersModel

# Undispatched: architecture on meta, no real weights.
model = TransformersModel("openai-community/gpt2", task="text-generation")

with model.scan("The Eiffel Tower is in"):
    dim = nnsight.save(model.transformer.h[0].output.shape[-1])   # int
    n_layers = nnsight.save(len(model.transformer.h))             # int
    hs = model.transformer.h[-1].output.save()                    # FakeTensor

print(dim, tuple(hs.shape))            # 768 (1, 7, 768)
print(model.dispatched)                # False — still not loaded
```

Shapes from a scan match the same reads under `model.trace(...)`. Modules must still be read in forward-pass order within an invoke.

## What a scan cannot tell you

A scan checks shapes. It is not a dry run of your intervention, and three things pass it that a real trace will not.

**Devices, on an undispatched model.** Unloaded weights live on `meta`, so every activation reports `device='meta'`, and a real CPU tensor is allowed to combine with a fake one. A steering vector on the wrong device passes:

```python
model = TransformersModel("openai-community/gpt2", device="cuda")       # undispatched
steering = torch.randn(768)                                             # CPU

with model.scan("The Eiffel Tower is in"):
    model.transformer.h[5].output[:, -1, :] += steering    # passes
```

Under `model.trace(...)` the same lines raise `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`. Scanning a dispatched model does catch it (`FakeTensorDeviceMismatchError`), but the point of a scan is usually that nothing is loaded. Derive `device=` from the activation you combine with.

**Real numbers.** `.item()` on a fake tensor returns a `torch.SymFloat` symbol, and arithmetic on it keeps propagating symbols; only the conversion to a real number raises, wherever a threshold or `float()` finally lands:

```python
with model.scan("The Eiffel Tower is in"):
    total = nnsight.save(model.transformer.h[0].output.sum().item())

print(repr(total + 1))  # zuf0 + 1.0
float(total)            # GuardOnDataDependentSymNode
```

**Anything about the values.** Branching on fake content raises `GuardOnDataDependentSymNode: Could not guard on data-dependent expression ...`. Branch on shapes and dtypes, which are real values; for content-dependent branching run a `trace`:

```python
with model.scan("Hello world"):
    out = model.transformer.h[0].output
    if out.shape[-1] > 1000:          # fine — an int
        wide = nnsight.save(True)
    # if (out > 0).all(): ...         # raises under scan
```

A saved `FakeTensor` stays usable after the scan — that is the trap. `.shape`, `.sum()`, `.cpu()`, `.tolist()` keep handing back fake tensors; `float(...)` raises `GuardOnDataDependentSymNode`, `.numpy()` raises, and mixing it with a real tensor raises `FakeTensorDeviceMismatchError`. Save the *shape*, not the fake tensor.

Only `scan` hands out fake tensors; inside `model.trace(...)` values are the real tensors the modules produced ([access-and-modify](access-and-modify.md)).

## Gotchas

- Some ops lack a fake/meta kernel and raise inside scan even though they work for real. Move that code out of the scan, or add the meta kernel upstream.

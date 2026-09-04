---
title: Scan
one_liner: Validate shapes/operations under FakeTensor mode without running the real model (`model.scan(...)`).
tags: [usage, scan, validation]
related: [docs/usage/trace.md, docs/usage/save.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/modeling/mixins/meta.py]
---

# Scan

## What this is for

`model.scan(input)` opens a tracing context that runs the model under PyTorch's `FakeTensorMode`. Tensors carry shape and dtype but no real data; no kernels run and — crucially — **the model is not dispatched**. Use it to validate shape-dependent code (slicing, reshapes, intervention indexing) or inspect activation shapes without loading weights.

It is an `InterleavingTracer` subclass — `ScanningTracer` (`src/nnsight/intervention/tracer.py`) — so all the same primitives (`.output`, `.input`, `.save()`, `tracer.invoke`, `tracer.cache`, ...) work; only the execution runs under fake tensors.

## When to use / when not to use

- Use to inspect tensor shapes without paying to dispatch/run the model.
- Use to catch index-out-of-range or shape-mismatch errors early.
- **Do not** use to compute real values — outputs are `FakeTensor`s with no data.
- Scanning does **not** dispatch the model — great for shape inspection on a meta-loaded (undispatched) model.
- Read [What a scan cannot tell you](#what-a-scan-cannot-tell-you) before you treat a passing scan as a green light.

## Canonical pattern

```python
import nnsight
from nnsight.modeling.transformers import TransformersModel

# Undispatched: architecture on meta, no real weights.
model = TransformersModel("openai-community/gpt2", task="text-generation")
print(model.dispatched)   # False

with model.scan("The Eiffel Tower is in"):
    dim = nnsight.save(model.transformer.h[0].output.shape[-1])   # int
    hs = model.transformer.h[-1].output.save()                    # FakeTensor

print(dim, tuple(hs.shape))            # 768 (1, 7, 768)
print("Fake" in type(hs).__name__)     # True
print(model.dispatched)                # False — still not loaded
```

## Why `.save()` is still required inside scan

Scan is a tracing context — it goes through the same save-gated push as `model.trace`. Only values you mark with `.save()` / `nnsight.save(...)` survive past the boundary (see `docs/usage/save.md`). Use `nnsight.save(...)` for non-tensor values (ints, lists):

```python
import nnsight

with model.scan("Hello") as tracer:
    shape_int = nnsight.save(model.transformer.h[0].output.shape[-1])
    n_layers = nnsight.save(len(model.transformer.h))
```

## Fidelity

Shapes seen in a fake scan match a real forward pass:

```python
with model.scan("The Eiffel Tower is in"):
    scanned = model.transformer.h[-1].output.save()
# tuple(scanned.shape) matches the same read under model.trace(...)
```

## How it works

`ScanningTracer.execute` defers to `InterleavingTracer.execute` (so a string prompt is still tokenized and invokes are still batched) but wraps it in a `FakeTensorMode`:

```python
with FakeTensorMode(
    allow_non_fake_inputs=True,
    shape_env=ShapeEnv(assume_static_by_default=True),
):
    super().execute(code)
```

`allow_non_fake_inputs=True` lets the meta-device parameters take part without being faked first; `assume_static_by_default=True` keeps shapes concrete rather than symbolic. The meta mixin (`Meta.interleave`, `src/nnsight/modeling/mixins/meta.py`) sees the active fake mode and **skips dispatch**, leaving parameters on the meta device.

## What a scan cannot tell you

A scan checks shapes. It is not a dry run of your intervention, and three things pass it that
a real trace will not.

**Devices, on an undispatched model.** Weights that have not been loaded live on `meta`, so
every activation a scan hands you reports `device='meta'` — and `FakeTensorMode` is
constructed with `allow_non_fake_inputs=True`, which deliberately lets a real CPU tensor
combine with a fake one. A steering vector on the wrong device is a common intervention bug,
and the scan that was supposed to catch it says nothing:

```python
model = TransformersModel("openai-community/gpt2", device_map="cuda")   # undispatched
steering = torch.randn(768)                                             # CPU

with model.scan("The Eiffel Tower is in"):
    model.transformer.h[5].output[:, -1, :] += steering    # passes
```

The same two lines under `model.trace(...)` on a dispatched model raise
`RuntimeError: Expected all tensors to be on the same device, but found at least two devices,
cuda:0 and cpu!`. Scanning a model that *has* been dispatched does catch it — the activations
then carry the real device and you get a `FakeTensorDeviceMismatchError` — but the whole point
of a scan is usually that nothing has been loaded yet. Derive `device=` from the activation
you are combining with and the question never arises.

**Real numbers.** `.item()` on a fake tensor returns a `torch.SymFloat` — a symbol, not a
number — and arithmetic on it keeps propagating symbols without complaint:

```python
with model.scan("The Eiffel Tower is in"):
    total = nnsight.save(model.transformer.h[0].output.sum().item())

print(repr(total))      # zuf0
print(repr(total + 1))  # zuf0 + 1.0
float(total)            # GuardOnDataDependentSymNode
```

Only the conversion to a real number raises. A threshold, a comparison, or a `float()` deep in
a helper is where the error finally lands.

**Anything about the values.** Branching on fake content
(`if (out > 0).all():`) raises `GuardOnDataDependentSymNode`; see
`docs/gotchas/types-and-values.md`.

A `FakeTensor` you save out of the block does not become invalid when the scan exits — that is
the trap. `.shape`, `.dtype`, `.sum()`, `.mean() + 1`, `.cpu()`, `.tolist()` all keep working
and keep handing back fake tensors. What fails is anything that needs a real number:
`float(hs.sum())` raises `GuardOnDataDependentSymNode`, `.numpy()` raises, and mixing it with
a real tensor raises `FakeTensorDeviceMismatchError`. Save the *shape*, not the fake tensor.

## Gotchas

- Outputs are `FakeTensor`s — you can read `.shape`/`.dtype`/`.device`, not data.
- `.save()` is required just like in `model.trace(...)`. Use `nnsight.save(...)` for non-tensor values.
- Access modules in forward-pass order within an invoke (same rule as trace) or hit `OutOfOrderError`.
- Some ops lack a fake/meta kernel and will raise inside scan even if they work in real mode. Move that code out of scan, or add the meta kernel upstream.

## Related

- `docs/usage/trace.md`
- `docs/usage/save.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/cache.md`

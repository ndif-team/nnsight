---
title: Scan
one_liner: Validate shapes/operations under FakeTensor mode without running the real model (`model.scan(...)`).
tags: [usage, scan, validation]
related: [docs/usage/trace.md, docs/usage/save.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracing/tracer.py:613, src/nnsight/intervention/envoy.py:282]
---

# Scan

## What this is for

`model.scan(input)` opens a tracing context that runs the model under PyTorch's `FakeTensorMode`. Tensors carry shape and dtype but no real data; no GPU memory is allocated and no kernels run. The point is to validate shape-dependent code (slicing, reshapes, intervention indexing) before paying the cost of a real forward pass.

It is an `InterleavingTracer` subclass — `ScanningTracer` (`src/nnsight/intervention/tracing/tracer.py:613`) — so all the same intervention primitives (`.output`, `.input`, `.save()`, `tracer.invoke`, `tracer.cache`, etc.) are available, only the execution backend swaps to fake-mode.

## When to use / when not to use

- Use to inspect tensor shapes when you cannot afford to dispatch / run the model.
- Use to catch index-out-of-range or shape-mismatch errors early.
- **Do not** use to compute real values — outputs are `FakeTensor`s with no data.
- Does not auto-dispatch the model — useful for shape inspection on meta-loaded models. See `MetaMixin.interleave` (`src/nnsight/modeling/mixins/meta.py:97`) which skips dispatch when `tracer` is a `ScanningTracer`.

## Canonical pattern

```python
import nnsight

with model.scan("Hello"):
    dim = nnsight.save(model.transformer.h[0].output[0].shape[-1])
    # Index validation: this raises if the tensor's last dim < 11
    model.transformer.h[0].output[0][:, 10] = 0

print(dim)  # e.g. 768
```

## Why `.save()` is still required inside scan

Scan is a tracing context — it goes through the same `Tracer.__exit__` → `Tracer.push()` path as `model.trace`. `Globals.stack` is incremented on entry and decremented on exit; only ids in `Globals.saves` survive across the boundary (see `docs/usage/save.md`).

Use `nnsight.save(...)` for non-tensor values:

```python
import nnsight

with model.scan("Hello"):
    shape_int = nnsight.save(model.transformer.h[0].output[0].shape[-1])  # int
    n_layers = nnsight.save(len(model.transformer.h))                      # int
    last_logits = model.lm_head.output[:, -1].save()                       # FakeTensor

print(shape_int, n_layers, last_logits.shape)
```

## Inspecting all module shapes

```python
shapes = {}
with model.scan("Hello") as tracer:
    cache = tracer.cache()  # FakeTensors fill the cache

# Inspect after exit
for path, entry in cache.items():
    if entry.output is not None and hasattr(entry.output, "shape"):
        shapes[path] = entry.output.shape
```

## How it works

`ScanningTracer.execute` wraps `super().execute` in `FakeTensorMode + FakeCopyMode` (`src/nnsight/intervention/tracing/tracer.py:621`):

```python
with FakeTensorMode(
    allow_non_fake_inputs=True,
    shape_env=ShapeEnv(assume_static_by_default=True),
) as fake_mode:
    with FakeCopyMode(fake_mode):
        self.batcher.batched_args = copy.deepcopy(self.batcher.batched_args)
        self.batcher.batched_kwargs = copy.deepcopy(self.batcher.batched_kwargs)
        super().execute(fn)
```

`allow_non_fake_inputs=True` means the user's batched inputs (real tensors / strings) are not turned into fakes; the model's parameters are auto-faked by `FakeCopyMode`. `assume_static_by_default=True` keeps shapes concrete numbers rather than symbolic.

## Skip dispatch

`MetaMixin.interleave` checks `isinstance(self.interleaver.tracer, ScanningTracer)` and **does not** auto-dispatch the model when scanning. This makes scan cheap on meta-loaded `LanguageModel`s.

## Gotchas

- Outputs are `FakeTensor`s. You cannot read their data — only shape, dtype, device.
- `.save()` is required just like in `model.trace(...)`.
- For non-tensor values (ints, lists, dicts), use `nnsight.save(...)` since `obj.save()` only works on objects pymount has injected onto.
- `nnsight.bool` is patched globally so `bool(fake_tensor)` returns `True` (`src/nnsight/__init__.py:128`) — this prevents some early-exit failures inside fake mode.
- Some operations are not supported by FakeTensor and will raise inside scan even if they work in real mode. Either add the corresponding fake meta kernel upstream or move that code out of scan.

## Related

- `docs/usage/trace.md`
- `docs/usage/save.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/cache.md`

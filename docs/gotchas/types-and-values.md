---
title: Types and Values Pitfalls
one_liner: Misconceptions about what's "inside" a trace — values are real tensors, not proxies; FakeTensor in scan behaves slightly differently; device placement matters.
tags: [gotcha, types, scan, faketensor, device]
related: [docs/concepts/deferred-execution.md, docs/usage/scan.md]
sources: [src/nnsight/__init__.py:128, src/nnsight/intervention/tracing/tracer.py:613, src/nnsight/intervention/interleaver.py:264]
---

# Types and Values Pitfalls

## TL;DR
- Inside a trace, `.output`/`.input` deliver *real* tensors. `print(...)`, `.shape`, and arithmetic all work directly. There are no proxies.
- Use `model.scan(input)` to inspect shapes and validate operations *without* running the model — values arrive as `FakeTensor`s.
- Tensors you create inside a trace must be put on the right device, e.g. `torch.randn(...).to(model.transformer.h[0].output[0].device)` — the model's tensors are on whatever device map you loaded with.
- Inside `.scan()`, `FakeTensor.__bool__` is patched to always return `True` (`src/nnsight/__init__.py:128`). Python `if` on a fake tensor does not reflect runtime truthiness.

---

## "Values inside a trace are proxies" — they are not

### Symptom
Code that "looks like it should work" works. Or: someone reads about deferred execution, assumes values are proxies, and writes code defensively (cloning everything, calling `.value` on things, treating shapes as opaque) when none of that is needed.

### Cause
nnsight's threading model is value-passing, not proxy-passing. When the worker thread accesses `.output`, it issues a `request(...)` call (`src/nnsight/intervention/interleaver.py:264`) that *blocks* until the model's hook hands back the actual tensor. The variable you receive is a real `torch.Tensor` (or whatever type the module returned). All standard PyTorch operations work on it directly.

### Wrong assumption (no error, but unnecessary code)
```python
with model.trace("Hello"):
    hs = model.transformer.h[0].output[0]
    shape = hs.shape   # already a real torch.Size, no .resolve() needed
    print(shape)
    zeros = torch.zeros(shape)   # works directly
    mean = hs.mean()             # works directly
```

### Right code (the same, framed correctly)
```python
with model.trace("Hello"):
    hs = model.transformer.h[0].output[0]
    print(hs.shape)              # torch.Size([1, 5, 768])
    print(hs.mean())             # real scalar
    print(hs.dtype, hs.device)   # all real attributes
```

### Mitigation / how to spot it early
- If you find yourself writing wrappers to "extract" or "resolve" values inside a trace, you're treating them as proxies — they're not. Just use them directly.
- If you need a real Python literal *outside* the trace (e.g. an int extracted from `.shape`), `.save()` it (or `nnsight.save(...)` for non-tensor types).

---

## Inspect shapes without running the model: use `.scan()`

### Symptom
You want to know the output shape of a layer without paying the cost of a real forward pass, or you want to validate that an indexing operation will work before doing it for real.

### Cause
`model.scan(input)` runs the model under `FakeTensorMode` (`src/nnsight/intervention/tracing/tracer.py:613`). Modules execute symbolically — no real computation, but shapes propagate. You can read `.shape`, validate slicing (`output[0][:, 1000]` will error early if the dim is too small), and confirm tuple structure.

`FakeTensor`s are not real tensors but they implement most introspection operations. Arithmetic on them returns more fake tensors with propagated shapes.

### Right code
```python
import nnsight

with model.scan("Hello"):
    dim = nnsight.save(model.transformer.h[0].output[0].shape[-1])

print(dim)   # 768
```

### Mitigation / how to spot it early
- Use `.scan()` for shape introspection / static validation.
- Use `.trace()` for actual computation.
- See [docs/usage/scan.md](../usage/scan.md).

---

## Device placement for tensors created inside a trace

### Symptom
`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` when you create a noise / steering vector locally and add it to a model activation.

### Cause
The model's tensors live on whatever device map you loaded with (e.g. `device_map="auto"` may put different layers on different GPUs). A tensor you make with `torch.randn(...)` defaults to CPU. Mixing them errors.

### Wrong code
```python
steering = torch.randn(768)   # CPU

with model.trace("Hello"):
    model.transformer.h[10].output[0][:, -1, :] += steering   # device mismatch
```

### Right code
```python
with model.trace("Hello"):
    target = model.transformer.h[10].output[0]
    steering = torch.randn(768).to(target.device)
    target[:, -1, :] += steering
```

Or compute the steering vector on the right device upfront:

```python
device = next(model.parameters()).device   # rough proxy for "main" device
steering = torch.randn(768, device=device)

with model.trace("Hello"):
    model.transformer.h[10].output[0][:, -1, :] += steering
```

### Mitigation / how to spot it early
- Read the target tensor's `.device` attribute and `.to(...)` your tensor onto it before the operation.
- For `device_map="auto"`, different layers may be on different devices — match each to its target.

---

## `FakeTensor.__bool__` always returns `True` in scan

### Symptom
Inside `model.scan(...)`, you write `if some_tensor:` or `if shape == 0:` and the branch always takes the "True" path regardless of the actual content.

### Cause
nnsight patches `FakeTensor.__bool__` to return `True` unconditionally (`src/nnsight/__init__.py:128`). Without this patch, many control-flow operations inside the model's forward pass would raise (because `FakeTensor.__bool__` would fail under `FakeTensorMode`'s symbolic execution). The patch lets the forward pass run to completion, but it also means *your* `if`-statements on fake tensors are not informative.

### Wrong assumption
```python
import nnsight

with model.scan("Hello"):
    out = model.transformer.h[0].output[0]
    if (out > 0).all():           # always True under fake mode
        print("non-negative")
    if out.shape[-1] > 1000:      # this is on torch.Size — works correctly
        print("wide hidden")
```

### Mental fix
Boolean operations *on shapes* (`torch.Size`, `int`) work normally — those aren't `FakeTensor`s. Boolean operations *on the fake tensor data* always return `True`.

For real branching on tensor *content*, you need `.trace()`, not `.scan()`.

### Mitigation / how to spot it early
- Inside `.scan()`, only branch on shape or dtype, never on tensor content.
- For runtime-content branching, run a real `.trace()`.

---

## Related
- [docs/concepts/deferred-execution.md](../concepts/deferred-execution.md) — full mental model of how interventions and the model interleave.
- [docs/usage/scan.md](../usage/scan.md) — scan reference.
- [docs/gotchas/save.md](save.md) — `.save()` is the only way to bring values out of any tracing context (including scan).

---
title: Types and Values Pitfalls
one_liner: Values inside a trace are real tensors, not proxies; scan gives FakeTensors (branching on their content raises); device placement matters.
tags: [gotcha, types, scan, faketensor, device]
related: [docs/concepts/deferred-execution.md, docs/usage/scan.md]
sources: [src/nnsight/intervention/interleaver.py:270, src/nnsight/intervention/tracer.py:299, src/nnsight/tracing/hint.py]
---

# Types and Values Pitfalls

## TL;DR
- Inside a `trace`, `.output`/`.input` deliver **real** tensors. `print`, `.shape`, `.mean()`, arithmetic all work directly. There are no proxies to "resolve".
- `model.scan(input)` runs the forward under `FakeTensorMode` — values come back as **`FakeTensor`s** carrying shape/dtype only. Read `.shape`/`.dtype`; a fake tensor is invalid once the scan exits.
- **Branching on fake-tensor *content* under scan raises** (`GuardOnDataDependentSymNode`), it does not silently return `True`. The old "`FakeTensor.__bool__` always True" behavior is gone. For content-dependent branching use `trace`, not `scan`.
- Branching on **shapes/ints** (`torch.Size`, `int`) works normally in scan — those aren't fake tensors.
- Tensors you create inside a trace must be on the model's device. Inputs *you pass to `trace`* are moved for you; tensors you build in the block are not.

---

## Values inside a trace are real, not proxies

### Symptom
Defensive code — cloning everything, calling `.value`, treating shapes as opaque — that isn't needed. Or surprise that ordinary tensor ops just work.

### Cause
Interleaving is value-passing, not proxy-passing. Reading `.output` parks the worker until the model produces the value, then hands back the **actual** object (`Mediator.value`, `src/nnsight/intervention/interleaver.py:270`) — a real `torch.Tensor` (or tuple, ...). It behaves as nnsight's `Object` tensor stand-in only for editor type hints; at runtime it *is* the tensor.

### Right code
```python
with model.trace("Hello world"):
    hs = model.transformer.h[0].output
    print(type(hs).__name__)     # Tensor
    print(hs.shape)              # torch.Size([1, 2, 768])
    print(hs.mean())             # a real scalar
    print(hs.dtype, hs.device)   # real attributes
```

### Mitigation
- If you're writing wrappers to "extract"/"resolve" values inside a trace, stop — use them directly.
- To get a plain Python value *outside* the trace, `.save()` it (or `nnsight.save(...)` for non-tensors).

---

## Inspect shapes without running the model: `.scan()`

### Cause
`model.scan(input)` runs the forward inside `FakeTensorMode` (`src/nnsight/intervention/tracer.py:299`): modules execute symbolically, propagating shape/dtype but doing no real compute and needing no real weights.

### Right code
```python
import nnsight

with model.scan("Hello world"):
    t = model.transformer.h[0].output
    dim = nnsight.save(t.shape[-1])          # 768
    kind = nnsight.save(type(t).__name__)    # 'FakeTensor'
print(dim, kind)
```

### Mitigation
- Use `scan` for shape introspection / static validation; `trace` for real computation.
- A fake tensor `.save()`d out of a scan is not usable once the fake mode exits — save the *shape*, not the fake tensor.

---

## Branching on fake-tensor content raises under scan

### Symptom
```
GuardOnDataDependentSymNode: Could not guard on data-dependent expression ...
```
when you write `if (out > 0).all():` (or `bool(...)`) on a fake tensor inside `scan`.

### Cause
There is no `FakeTensor.__bool__` override in the current nnsight. Torch's fake mode cannot decide a data-dependent boolean symbolically, so it raises. (This is unlike older nnsight, which patched `__bool__` to always return `True`.)

### Right code
```python
import nnsight

with model.scan("Hello world"):
    out = model.transformer.h[0].output
    if out.shape[-1] > 1000:          # OK — this is a torch.Size / int
        wide = nnsight.save(True)
    # if (out > 0).all(): ...         # raises — content-dependent under fake mode
```

### Mitigation
- Inside `scan`, branch only on shape/dtype. For branching on tensor *content*, run a real `trace`.

---

## Device placement for tensors created inside a trace

### Symptom
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

### Cause
`interleave` moves the inputs *you pass to `trace`* onto the model's device (`src/nnsight/intervention/envoy.py`), but a tensor you construct inside the block (`torch.randn(...)`) defaults to CPU. Adding it to a GPU activation errors.

### Wrong / Right
```python
# wrong — steering vector is on CPU
steering = torch.randn(768)
with model.trace("Hello world"):
    model.transformer.h[10].output[:, -1, :] += steering

# right — move onto the target's device
with model.trace("Hello world"):
    target = model.transformer.h[10].output
    steering = torch.randn(768).to(target.device)
    target[:, -1, :] += steering
```

### Mitigation
- Read the target's `.device` and `.to(...)` your tensor onto it. With `device_map="auto"`, different layers can be on different devices — match each.

---

## Related
- [docs/concepts/deferred-execution.md](../concepts/deferred-execution.md) — how interventions interleave with the model.
- [docs/usage/scan.md](../usage/scan.md) — scan reference.
- [docs/gotchas/save.md](save.md) — getting values out of any tracing context.

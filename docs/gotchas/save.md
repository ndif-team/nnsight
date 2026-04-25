---
title: Save Pitfalls
one_liner: Anything that goes wrong with .save() / nnsight.save() — values disappearing, remote returns, scan persistence, pymount edge cases.
tags: [gotcha, save, remote]
related: [docs/usage/save.md, docs/gotchas/remote.md]
sources: [src/nnsight/intervention/tracing/globals.py:9, src/nnsight/intervention/tracing/base.py:537, src/nnsight/__init__.py:78]
---

# Save Pitfalls

## TL;DR
- Forget `.save()` and the variable is filtered out when the trace exits — you get `NameError` or stale references.
- `.save()` is required inside `model.scan(...)` too, not only `model.trace(...)`.
- For remote traces, `.save()` is the *only* mechanism that transmits values back — non-saved values are dropped on the server.
- `local_list.append(x.save())` where `local_list` was defined *outside* the trace ends up empty for remote traces; create the list *inside* the trace.
- Prefer `nnsight.save(x)` over `x.save()` for non-tensor values — it does not rely on the `PYMOUNT` C extension that monkey-patches `object.save`.

---

## Forgetting `.save()`

### Symptom
After the `with model.trace(...)` block exits, the variable you tried to inspect either no longer exists or only has its pre-trace value. Often appears as `NameError: name 'output' is not defined` or as a tensor that "looks right inside the trace but is empty outside".

### Cause
On exit, the tracer filters the trace frame's local variables. When the global `Globals.stack == 1` (i.e. the outermost trace context is exiting), only variables whose `id(...)` is in the `Globals.saves` set get pushed back to the user's calling frame (see `src/nnsight/intervention/tracing/base.py:537`). `.save()` and `nnsight.save()` are exactly the calls that add the object's id to that set (`src/nnsight/intervention/tracing/globals.py:9`).

### Wrong code
```python
with model.trace("Hello"):
    output = model.transformer.h[-1].output[0]   # not saved
print(output)   # NameError or stale
```

### Right code
```python
with model.trace("Hello"):
    output = model.transformer.h[-1].output[0].save()
print(output.shape)   # torch.Size([1, 2, 768])
```

### Mitigation / how to spot it early
- If a variable "exists inside the trace but disappears outside", you forgot `.save()`.
- Make `.save()` your default — strip it back if you don't need the value.

---

## `.save()` inside `model.scan(...)`

### Symptom
Calling `model.scan("Hello")` to inspect shapes, then trying to read the result outside the scan block: `NameError`, or you get a stale value from before the scan.

### Cause
`model.scan(...)` is itself a tracing context (it pushes onto `Globals.stack`). The same exit-filter logic applies — non-saved local variables are dropped. Plain `.shape` returns a `torch.Size` (or an `int` indexed off it), neither of which has `Object.save` mounted on it unless `PYMOUNT` is on, so use `nnsight.save(...)`.

### Wrong code
```python
with model.scan("Hello"):
    dim = model.transformer.h[0].output[0].shape[-1]
print(dim)   # NameError
```

### Right code
```python
import nnsight

with model.scan("Hello"):
    dim = nnsight.save(model.transformer.h[0].output[0].shape[-1])
print(dim)   # 768
```

### Mitigation / how to spot it early
- If the symptom is "`.scan()` works inside the block but I lose the value outside", that's this gotcha.
- See [docs/usage/scan.md](../usage/scan.md) for the full scan doc.

---

## Remote `.save()` is the transmission channel

### Symptom
A remote trace (`remote=True`) completes, the request succeeds, but the variable you wanted comes back as `None`, missing, or unchanged from its pre-trace value.

### Cause
For remote execution the worker process serializes only the saved variables back to the client. `.save()` is the explicit "ship this value home" marker — without it, the server discards the value when the trace block ends.

### Wrong code
```python
with model.trace("Hello", remote=True):
    output = model.lm_head.output    # not saved
print(output)   # nothing useful comes back
```

### Right code
```python
with model.trace("Hello", remote=True):
    output = model.lm_head.output.detach().cpu().save()
print(output.shape)
```

### Mitigation / how to spot it early
- For remote, *always* `.save()` the values you want back. Move tensors to CPU first to keep the payload small.
- See [docs/gotchas/remote.md](remote.md) for related remote pitfalls.

---

## Local list `.append(x.save())` with a list created *outside* the trace

### Symptom
For remote traces, you collect values into a list across loop iterations. The remote run completes successfully, but the list ends up empty or contains only pre-trace placeholders.

### Cause
A list created *outside* the remote trace never travels to the server — only the data sent with the trace request goes. Calls to `.append(...)` happen on the server's heap, not the client's. When the request returns, the local list is still its original empty state.

The fix is to create the list *inside* the trace and call `.save()` on the list itself, so the populated list is what comes back.

### Wrong code
```python
captured = []   # lives only on the client
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    for step in tracer.iter[:]:
        captured.append(model.lm_head.output.argmax(dim=-1))   # appends server-side
print(captured)   # still []
```

### Right code
```python
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    captured = list().save()   # created server-side, marked for return
    for step in tracer.iter[:]:
        captured.append(model.lm_head.output.argmax(dim=-1))
print(captured)
```

### Mitigation / how to spot it early
- Rule of thumb for remote: any container you intend to populate during the trace must be created inside the trace and `.save()`-ed.
- Local-only traces tolerate either pattern because the list and the trace frame live in the same process.

---

## `nnsight.save(x)` vs `x.save()`

### Symptom
`x.save()` errors with `AttributeError: 'list' object has no attribute 'save'` (or similar) when `CONFIG.APP.PYMOUNT = False`, or when `x` is an object whose class defines its own `.save` method that shadows nnsight's mounted one.

### Cause
`x.save()` works in two different ways depending on the type of `x`:

- For tensors, `.save()` is provided by nnsight's `Object` torch.Tensor subclass.
- For arbitrary Python objects (lists, dicts, ints, etc.), `.save()` only exists because nnsight uses a C extension (`pymount`) to inject the method onto the base `object` class at runtime. This is gated by `CONFIG.APP.PYMOUNT` (default `True`). See `src/nnsight/intervention/tracing/globals.py:104`.

If pymount is disabled, or the object's own class defines a different `.save` that shadows the mounted one (e.g. an HF model whose `.save_pretrained` aliases `.save`), the call dispatches to something that isn't nnsight's save, and the value is not registered with `Globals.saves`.

`nnsight.save(x)` (`src/nnsight/intervention/tracing/globals.py:9`) is just a function that adds `id(x)` to the saves set and returns `x`. It works on every object, has no monkey-patching dependency, and never collides with the object's own methods.

### Wrong code
```python
import nnsight
nnsight.CONFIG.APP.PYMOUNT = False   # imagine a deployment where pymount is disabled

with model.trace("Hello"):
    shape = model.transformer.h[0].output[0].shape.save()   # AttributeError
```

### Right code
```python
import nnsight

with model.trace("Hello"):
    shape = nnsight.save(model.transformer.h[0].output[0].shape)
print(shape)
```

### Mitigation / how to spot it early
- For non-tensor values (shapes, ints, lists, dicts), use `nnsight.save(...)` unconditionally. It's safer and the cost is a single function call.
- For tensors, both forms are equivalent in behavior — `nnsight.save(t)` is still a fine default.

---

## Related
- [docs/usage/save.md](../usage/save.md) — full `.save()` semantics.
- [docs/gotchas/remote.md](remote.md) — more remote-specific pitfalls.
- [docs/usage/scan.md](../usage/scan.md) — scan context details.

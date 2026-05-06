---
title: Save Pitfalls
one_liner: Anything that goes wrong with .save() / nnsight.save() — values disappearing, remote returns, scan persistence, pymount edge cases.
tags: [gotcha, save, remote]
related: [docs/usage/save.md, docs/gotchas/remote.md]
sources: [src/nnsight/intervention/tracing/globals.py:9, src/nnsight/intervention/tracing/base.py:537, src/nnsight/__init__.py:78]
---

# Save Pitfalls

## TL;DR
- Forget `.save()` and the variable is filtered out when the **root** trace exits — you get `NameError` or stale references.
- `.save()` is required inside `model.scan(...)` too, not only `model.trace(...)`.
- **Reassigning a saved name to an unsaved value un-saves it.** `x = t.save(); x = 2` leaves the outer `x` at its pre-trace value — the save filter tracks `id(value)`, not the name.
- **Nested traces don't need `.save()` between them.** Variables flow freely between an inner trace and its enclosing tracing context (e.g. inside a `model.session()`). Only the root-trace boundary applies the save filter.
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
    output = model.transformer.h[-1].output   # not saved
print(output)   # NameError or stale
```

### Right code
```python
with model.trace("Hello"):
    output = model.transformer.h[-1].output.save()
print(output.shape)   # torch.Size([1, 2, 768])
```

### Mitigation / how to spot it early
- If a variable "exists inside the trace but disappears outside", you forgot `.save()`.
- Make `.save()` your default — strip it back if you don't need the value.

---

## Reassigning a saved name un-saves it

### Symptom
You called `.save()` on a value, then later in the same trace rebound the same name to something unsaved. After the trace exits, the outer name still holds its pre-trace value (or is undefined) — the saved tensor is lost.

### Cause
The save filter tracks the `id(...)` of the saved object, not the name it was bound to. Rebinding the name to a new (unsaved) object means the saved tensor is no longer reachable through that local at trace-exit time, and the filter drops everything else.

### Wrong code
```python
x = "default"
with model.trace("a"):
    x = torch.tensor([1, 2, 3]).save()
    x = 2                            # rebinds x to an unsaved int
print(x)   # "default" (or NameError if x was undefined before the trace)
```

### Right code
```python
with model.trace("a"):
    x = torch.tensor([1, 2, 3]).save()   # leave x bound to the saved tensor
print(x)   # tensor([1, 2, 3])
```

If you need the integer too, give it its own name and save it explicitly:
```python
import nnsight
with model.trace("a"):
    x = torch.tensor([1, 2, 3]).save()
    y = nnsight.save(2)
print(x, y)
```

### Mitigation / how to spot it early
- Treat `.save()` as binding the *value* to its return slot; don't reuse the name for something else in the same trace.
- If a saved tensor mysteriously disappears, search the trace body for any later assignment to the same name.

---

## Nested traces don't need `.save()` between them

### Symptom
You're inside a `model.session()` (or any outer tracing context) and want to use a variable produced inside an inner `model.trace()` from another inner trace. You're not sure whether to `.save()` it.

### Cause
The save filter runs **only when the root tracing context exits**. While `Globals.stack > 1` (i.e., you are inside one tracing context that is itself inside another), the inner context's `push` keeps every variable, not just saved ones (`src/nnsight/intervention/tracing/base.py:537`).

The check is literally:

```python
if Globals.stack == 1:        # only the outermost trace filters
    filtered_state = {
        k: v for k, v in filtered_state.items() if id(v) in Globals.saves
    }
    Globals.saves.clear()
```

So `.save()` only matters at the **root** trace boundary. For variables that need to cross the **outer** boundary (the session's `with` block exiting back to user code), `.save()` is required. For variables that just need to flow between sibling traces inside the session, `.save()` is unnecessary.

### Right code

```python
import nnsight

with model.session():
    # Inner trace 1: capture a value, no .save() needed inside the session
    with model.trace("Madison Square Garden is in the city of"):
        hs = model.transformer.h[5].output[:, -1, :]   # no .save()

    # Inner trace 2: use the value from the previous trace
    with model.trace("_ _ _ _ _ _ _"):
        model.transformer.h[5].output[:, -1, :] = hs   # works
        patched = model.lm_head.output[0][-1].argmax(dim=-1).save()  # SAVE — leaves the session

print(patched)  # available outside the session
```

`patched` needed `.save()` because it crosses the outermost tracing boundary back into ordinary Python. `hs` did not — it only crossed an inner-to-inner boundary.

### Right code — three traces in a session

```python
with model.session():
    with model.trace("Hello"):
        hs1 = model.transformer.h[0].output    # no save
    with model.trace("World"):
        hs2 = model.transformer.h[0].output    # no save
    with model.trace("Combined"):
        combined = (hs1 + hs2).save()             # save — leaves the session
```

### Mitigation / how to spot it early
- The rule is simple: **`.save()` at the root boundary only**. Anything you write to need outside the outermost `with` block.
- Inside any inner trace (a trace inside a session, a trace inside a session inside a session), variables propagate freely. Only the outermost exit filters.
- This is also true for remote sessions: only the outermost remote `with model.session(remote=True):` exit triggers the save filter on the server side.

### Related
- `docs/usage/sessions.md` — full session semantics.
- [docs/gotchas/cross-invoke.md](cross-invoke.md) — for cross-*invoke* (not cross-trace) variable propagation, which is governed by `CONFIG.APP.CROSS_INVOKER`.

---

## `.save()` inside `model.scan(...)`

### Symptom
Calling `model.scan("Hello")` to inspect shapes, then trying to read the result outside the scan block: `NameError`, or you get a stale value from before the scan.

### Cause
`model.scan(...)` is itself a tracing context (it pushes onto `Globals.stack`). The same exit-filter logic applies — non-saved local variables are dropped. Plain `.shape` returns a `torch.Size` (or an `int` indexed off it), neither of which has `Object.save` mounted on it unless `PYMOUNT` is on, so use `nnsight.save(...)`.

### Wrong code
```python
with model.scan("Hello"):
    dim = model.transformer.h[0].output.shape[-1]
print(dim)   # NameError
```

### Right code
```python
import nnsight

with model.scan("Hello"):
    dim = nnsight.save(model.transformer.h[0].output.shape[-1])
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
    shape = model.transformer.h[0].output.shape.save()   # AttributeError
```

### Right code
```python
import nnsight

with model.trace("Hello"):
    shape = nnsight.save(model.transformer.h[0].output.shape)
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

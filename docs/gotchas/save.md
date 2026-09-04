---
title: Save Pitfalls
one_liner: Anything that goes wrong with .save() / nnsight.save() — the outside-a-trace guard, values disappearing, aliasing, remote returns.
tags: [gotcha, save, remote]
related: [docs/usage/save.md, docs/errors/save-outside-trace.md, docs/gotchas/remote.md]
sources: [src/nnsight/tracing/tracer.py, src/nnsight/tracing/hint.py]
---

# Save Pitfalls

## TL;DR
- **`save()` raises outside a trace.** `nnsight.save(x)` / `x.save()` marks a value to return from the enclosing `with model.trace(...):` block, so calling it with no trace running is a `ValueError`. Move the save inside the block.
- Forget `.save()` and the variable does not cross back out of the trace: reading it afterward is a `NameError` in a script, an `UnboundLocalError` inside a function.
- `.save()` / `nnsight.save(x)` return `x` **unchanged** — save the value you bind: `h = module.output.save()`. A value built from a saved one is not itself saved: `(x.save() * 2)` returns `x`, write `(x * 2).save()`.
- **Nested traces don't need `.save()` between them.** Only the *outermost* trace boundary filters to saved values; inner traces (inside a `model.session()`) push everything up.
- `.save()` is required inside `model.scan(...)` too — it is a tracing context like `trace`.
- For remote traces, `.save()` is the *only* mechanism that transmits values back.
- `x.save()` works on any object via an optional C mount (`CONFIG.APP.PYMOUNT`, default on). If the extension isn't built, `x.save()` on a non-tensor raises `AttributeError` — use `nnsight.save(x)`, which never depends on the mount.

---

## `save()` outside a trace raises

### Symptom
```
ValueError: save() was called outside a trace. `.save()` / nnsight.save(x) marks a
value to return from the enclosing `with model.trace(...):` block, so it only works
inside one — move the save into the trace block.
```

### Cause
`save` marks an object (by identity) so the trace's exit knows to hand it back to the caller. With no trace running there is nothing to hand it back *from*, and the mark would be cleared before anything read it. `save` (`src/nnsight/tracing/tracer.py`) checks the per-thread trace depth and raises if it is zero.

### Wrong code
```python
import nnsight

captured = nnsight.save([])          # ValueError — no trace active yet
with model.trace("Hello"):
    captured.append(model.transformer.h[0].output[:, -1, :])
```

### Right code
```python
import nnsight

with model.trace("Hello"):
    captured = nnsight.save([])       # created and marked inside the trace
    captured.append(model.transformer.h[0].output[:, -1, :])
print(len(captured))
```

### Mitigation / how to spot it early
- Every `save()` call must be lexically inside a `with model.trace(...):` / `scan(...)` / `session(...)` block.
- To collect across steps, create the container inside the trace and save *it* (`captured = nnsight.save([])`), then append **raw** values — see [Collecting values across steps](#collecting-values-across-steps--invokes).

---

## Forgetting `.save()`

### Symptom
After the `with model.trace(...)` block exits, reading the variable raises. At module scope —
a script — that is `NameError: name 'output' is not defined`; inside a function it is
`UnboundLocalError: cannot access local variable 'output' where it is not associated with a
value`, because the assignment inside the block made the name local.

### Cause
The trace body runs in a scratch namespace; on exit only the values marked with `save` are written back into your frame (`push_result`, `src/nnsight/tracing/tracer.py`). An unsaved name was assigned inside the block but never pushed back, so referencing it afterward hits an unbound local.

### Wrong code
```python
with model.trace("Hello"):
    output = model.transformer.h[-1].output   # not saved
print(output)   # NameError in a script, UnboundLocalError in a function
```

### Right code
```python
with model.trace("Hello"):
    output = model.transformer.h[-1].output.save()
print(output.shape)   # torch.Size([1, 2, 768])
```

### Mitigation / how to spot it early
- If a variable "exists inside the trace but is unbound outside", you forgot `.save()`.
- Make `.save()` your default — strip it back when you don't need the value.

---

## Collecting values across steps / invokes

### Symptom
You want a list of intermediate values gathered during the trace, available afterward.

### Cause
`save` marks the object you bind to a name; on exit only those marked, named locals are pushed back to your frame. So the rule for a collection is: **save the container, put raw values in it.** Saving the individual elements instead (`xs.append(x.save())`) marks values with no name to return under, and leaving the container unsaved means it is never pushed back at all.

### Right code (saved container, raw values)
```python
import nnsight
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    steps = nnsight.save([])                  # save the list itself, inside the trace
    for _ in tracer.iter[:3]:
        steps.append(model.transformer.h[-1].output[:, -1, :])   # append raw values
print(len(steps))   # 3
```

A comprehension works the same way — save the whole list, keep the elements raw:
```python
with model.trace("Hello"):
    hiddens = nnsight.save([block.output for block in model.transformer.h])
```

### Wrong code
```python
# 1) Container assigned inside the trace but not saved -> unbound after.
with model.trace("Hello"):
    hiddens = [block.output.save() for block in model.transformer.h]   # not saved
print(len(hiddens))            # NameError / UnboundLocalError

# 2) Saving the elements into an outer list -> works locally by frame side effect,
#    but returns nothing on a remote trace (the appends happen server-side).
hiddens = []
with model.trace("Hello", remote=True):
    for block in model.transformer.h:
        hiddens.append(block.output.save())   # no name to return each under
print(len(hiddens))            # 0 on remote
```

### Mitigation / how to spot it early
- One rule: `xs = nnsight.save([])` inside the trace, then append/index **raw** values into `xs`. Never `.save()` the elements.
- Put the whole loop under a bounded `tracer.iter[:N]` (unbounded `iter[:]` drops trailing code — see [iteration.md](iteration.md)).

---

## Nested traces don't need `.save()` between them

### Symptom
Inside a `model.session()` you want a value produced in one inner `model.trace()` used by another. Unsure whether to `.save()` it.

### Cause
The save filter runs **only when the outermost tracing context exits** (`push_result` checks the trace depth is 1). Inner traces push *all* their locals up to the enclosing block, so variables flow freely between sibling traces in a session (not on `VLLM`, where each trace is its own request); only values crossing the outermost boundary back to plain Python are filtered.

### Right code
```python
import nnsight

with model.session():
    with model.trace("Madison Square Garden is in the city of"):
        hs = model.transformer.h[5].output[:, -1, :]        # no .save() needed
    with model.trace("_ _ _ _ _ _ _"):
        model.transformer.h[5].output[:, -1, :] = hs        # flows in
        patched = model.output.logits.argmax(dim=-1).save()  # SAVE — leaves the session
print(patched)
```

`patched` needed `.save()` because it crosses the outermost boundary; `hs` did not.

### Mitigation / how to spot it early
- The rule is: **`.save()` at the outermost boundary only.** This also holds for remote sessions (`with model.session(remote=True):`).

---

## `.save()` inside `model.scan(...)`

### Symptom
`model.scan("Hello")` to inspect shapes, then reading the result outside the block: the name
is not there.

### Cause
`model.scan(...)` is a tracing context like `trace` (it runs the forward under fake tensors). The same exit filter applies. Shapes come back as `torch.Size` / `int`, which have `.save()` mounted only when the C extension is built — so `nnsight.save(...)` is the safe form.

### Wrong / Right
```python
import nnsight

with model.scan("Hello"):
    dim = nnsight.save(model.transformer.h[0].output.shape[-1])
print(dim)   # 768
```

See [types-and-values.md](types-and-values.md) for scan value semantics.

---

## `nnsight.save(x)` vs `x.save()`

### Cause
`x.save()` on an arbitrary object exists only because nnsight optionally mounts a `save` method onto every object via a C extension, gated by `CONFIG.APP.PYMOUNT` (default `True`, `src/nnsight/__init__.py`). If the extension didn't build, the mount is silently skipped and `list().save()` / `some_size.save()` raise `AttributeError`. Tensors read from `.output`/`.input` always have `.save()` (they behave as nnsight's `Object` tensor stand-in).

`nnsight.save(x)` (`src/nnsight/tracing/tracer.py`) is a plain function that marks `id(x)` and returns `x`. It works on every object with no mount dependency.

### Mitigation
- For non-tensor values (shapes, ints, lists, dicts), prefer `nnsight.save(...)`.
- For tensors, both forms are equivalent.

---

## Related
- [docs/usage/save.md](../usage/save.md) — full `.save()` semantics.
- [docs/errors/save-outside-trace.md](../errors/save-outside-trace.md) — the outside-a-trace error.
- [docs/gotchas/remote.md](remote.md) — remote-specific save pitfalls.

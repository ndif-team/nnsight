---
title: Remote Execution Pitfalls
one_liner: NDIF-specific traps — .save() is the transmission channel, local containers don't get populated remotely, sessions vs traces, env mismatches.
tags: [gotcha, remote, ndif]
related: [docs/remote/index.md, docs/remote/ndif-overview.md, docs/gotchas/save.md]
sources: [src/nnsight/intervention/tracing/globals.py:9, src/nnsight/intervention/tracing/base.py:537]
---

# Remote Execution Pitfalls

## TL;DR
- `.save()` is the *only* channel that transmits values back from the NDIF server. Anything not saved is dropped.
- A list created *outside* the trace and `.append`-ed inside a remote trace ends up empty — the appends happen on the server. Build the list inside the trace.
- `.detach().cpu()` before `.save()` to keep download payloads small.
- Put `remote=True` on `model.session(...)`, not on the inner `model.trace(...)` calls.
- `print(...)` inside a remote trace shows up as `LOG` status during the request lifecycle, not in your local stdout.
- Mismatched local vs server environments raise warnings — see [docs/remote/env-comparison.md].
- Helper functions and classes defined locally must be registered (see [docs/remote/register-local-modules.md]) so the server can import them.

---

## `.save()` is the transmission channel

### Symptom
A remote request completes successfully, but the variable you wanted comes back as `None`, missing, or stuck at its pre-trace value. No error.

### Cause
The remote backend serializes only saved variables and ships them back to the client (see `Globals.saves` filtering in `src/nnsight/intervention/tracing/base.py:537`). Without `.save()`, the variable's `id` is never added to the saves set, so it's filtered out before the result is shipped.

### Wrong code
```python
with model.trace("Hello", remote=True):
    output = model.lm_head.output    # not saved
print(output)   # None / missing
```

### Right code
```python
with model.trace("Hello", remote=True):
    output = model.lm_head.output.detach().cpu().save()
print(output.shape)
```

### Mitigation / how to spot it early
- For remote, treat `.save()` as mandatory for every value you want returned.
- `.detach().cpu()` first reduces the payload — gradients and GPU tensors are heavy.

---

## Local list `.append` outside the trace

### Symptom
You build up a list of intermediate values across an iter loop or multiple `tracer.invoke(...)` blocks. Locally it works fine; remotely the list is empty.

### Cause
The list lives in the *client's* Python heap. `.append(...)` calls inside the remote trace happen in the *server's* Python heap, on the server's copy of that variable. Once the trace returns, the server discards its copy. The client's list was never modified.

The fix is to create the list *inside* the trace and call `.save()` on it. The server's populated list is what gets shipped back.

### Wrong code
```python
captured = []   # client-side
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    for step in tracer.iter[:]:
        captured.append(model.lm_head.output.argmax(dim=-1))
print(captured)   # []
```

### Right code
```python
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    captured = list().save()   # server-side, marked for return
    for step in tracer.iter[:]:
        captured.append(model.lm_head.output.argmax(dim=-1))
print(captured)
```

### Mitigation / how to spot it early
- Any container you populate inside a remote trace must be created inside it and `.save()`-ed.
- This is the same as the local case in principle — `.save()` is the only surviving variable channel — but the consequence is more visible remotely because of the process boundary.

---

## Move tensors to CPU before save

### Symptom
Remote downloads are slow, network costs are high, or you OOM the client when collecting many intermediate tensors.

### Cause
Tensors saved with `.save()` are serialized at their current device and dtype. GPU tensors, full-precision tensors, and tensors with autograd graphs attached are heavier than necessary for transport.

### Wrong code
```python
with model.trace("Hello", remote=True):
    hs = model.transformer.h[0].output.save()    # GPU tensor with grad info
```

### Right code
```python
with model.trace("Hello", remote=True):
    hs = model.transformer.h[0].output.detach().cpu().save()
```

### Mitigation / how to spot it early
- Make `.detach().cpu()` part of your remote-save habit.
- Cast to `bfloat16` or `float16` if precision allows further savings.

---

## `remote=True` on session, not on inner traces

### Symptom
You pass `remote=True` on each inner `model.trace(...)` inside a `model.session(...)` block. Each trace makes its own request. You wait in the queue multiple times. Variables don't flow between traces the way you expected.

### Cause
A session bundles multiple traces into a *single* remote request. Putting `remote=True` on the outer `model.session(...)` opts the entire session into remote execution; the inner traces are part of that one request. Putting `remote=True` on the inner traces creates separate requests, defeating the bundling.

### Wrong code
```python
# Each inner trace is its own request — multiple queue waits
with model.session():
    with model.trace("A", remote=True):
        a = ...save()
    with model.trace("B", remote=True):
        b = ...save()
```

### Right code
```python
# Single session-level request — values flow between traces directly
with model.session(remote=True):
    with model.trace("A"):
        a = model.transformer.h[5].output    # no .save() needed within session
    with model.trace("B"):
        model.transformer.h[5].output[:] = a
        result = model.lm_head.output.save()
```

### Mitigation / how to spot it early
- One `remote=True`, on the outermost context. Inner contexts inherit.
- Sessions also let intermediate variables flow between traces without explicit `.save()`.

---

## `print(...)` inside remote traces

### Symptom
You add `print(...)` calls inside a remote trace to debug. You don't see anything in your local terminal. The remote run still works.

### Cause
Remote traces capture stdout and ship `print` output as `LOG` status messages on the request lifecycle. They're visible in the request status stream (alongside `RECEIVED`, `QUEUED`, `RUNNING`, etc.), not in your local stdout.

### Wrong assumption
"`print` inside a remote trace will show up in my terminal."

### Right approach
- Watch the request status stream — `LOG` entries contain your prints.
- For repeated debugging, move the value out via `.save()` instead of `print` so you have the actual data after the run.
- Disable remote logging if you want quieter output: `CONFIG.APP.REMOTE_LOGGING = False`.

### Mitigation / how to spot it early
- If a remote run "looks silent", confirm `REMOTE_LOGGING` is on and watch the status stream.

---

## Environment mismatch warnings

### Symptom
Remote runs print warnings about Python version, PyTorch version, or nnsight version differing between your local environment and the NDIF server.

### Cause
Some objects (especially custom ones serialized by pickle/cloudpickle) are sensitive to library versions. Significant divergence can cause runs to fail or behave subtly differently. nnsight surfaces these mismatches as warnings so you can decide whether to upgrade.

### Mitigation
- Match your local versions to NDIF's whenever possible.
- See [docs/remote/env-comparison.md] for the full list of compared fields.

---

## Local helper functions and classes need registration

### Symptom
A remote run errors with `ModuleNotFoundError` or `AttributeError` for a function or class you defined in your local script — the server can't find it.

### Cause
nnsight ships intervention code as source plus a serialized closure. Anything referenced by name (helper functions, custom modules, dataclasses) needs to be importable on the server. Modules that exist only in your local script aren't.

`nnsight.register(module_or_callable)` (see [docs/remote/register-local-modules.md]) ships the source so the server can import it.

### Mitigation / how to spot it early
- If a remote trace works locally but fails on NDIF for missing symbols, register the local helpers.
- Standard library / well-known third-party modules (torch, numpy, transformers) don't need registration — they're already installed on the server.

---

## Related
- [docs/remote/index.md](../remote/index.md) — remote execution overview.
- [docs/remote/ndif-overview.md](../remote/ndif-overview.md) — NDIF lifecycle and request flow.
- [docs/gotchas/save.md](save.md) — `.save()` mechanics (which underlie all the above).
- [docs/gotchas/iteration.md](iteration.md) — `iter[:]` swallowing trailing code is amplified remotely (you might miss the warning in the status stream).

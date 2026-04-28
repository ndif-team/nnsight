---
title: Save
one_liner: Persist values from a tracing context with `nnsight.save(obj)` or `obj.save()`.
tags: [usage, tracing, save]
related: [docs/usage/trace.md, docs/usage/scan.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracing/globals.py:9, src/nnsight/intervention/tracing/globals.py:91, src/nnsight/intervention/tracing/base.py:537]
---

# Save

## What this is for

Inside a tracing context (`model.trace`, `model.generate`, `model.scan`, `model.session`, `tensor.backward`), variable assignments do **not** automatically escape the with-block. The tracing system only pushes back to the caller's frame the values explicitly marked as "saved". Marking a value as saved is done by `nnsight.save(value)` or, equivalently, `value.save()`.

## When to use / when not to use

- Use **always** — every value you want to read after a `with model.trace(...):` block exits.
- Required inside `model.scan(...)` too — it is a tracing context like the others. See `docs/usage/scan.md`.
- Not needed for tensors that you only read inside the body. Saved values cost you nothing extra unless they pin GPU memory you'd rather free.

## Canonical pattern

```python
import nnsight

with model.trace("Hello"):
    # PREFERRED: works on any object, no C-extension needed
    hidden = nnsight.save(model.transformer.h[-1].output)

    # ALSO WORKS: backwards-compatible method form
    logits = model.lm_head.output.save()

print(hidden.shape, logits.shape)
```

## Why `.save()` is needed

`Tracer.push()` (`src/nnsight/intervention/tracing/base.py:497`) is the bridge that copies locals from the worker frame back to the caller's frame on `__exit__`. When `Globals.stack == 1` (i.e. you are exiting the outermost trace), `push()` filters its candidate set to **only** ids in `Globals.saves` (`base.py:537`):

```python
if Globals.stack == 1:
    filtered_state = {
        k: v for k, v in filtered_state.items() if id(v) in Globals.saves
    }
    Globals.saves.clear()
```

`nnsight.save(obj)` simply does `Globals.saves.add(id(obj))` (`globals.py:9`).

## Two forms — prefer `nnsight.save()`

```python
import nnsight

# Function form — recommended
out = nnsight.save(model.transformer.h[0].output)

# Method form — backwards-compatible
out = model.transformer.h[0].output.save()
```

The method form depends on **pymount**, a C extension that monkey-patches a `.save` attribute onto every Python object at runtime. The function form does not. If a class defines its own `.save` method (e.g. `transformers.PreTrainedModel.save_pretrained` is unrelated, but third-party classes can shadow it), `obj.save()` will call that one instead of nnsight's. `nnsight.save()` is unaffected.

`Object.save` (`src/nnsight/intervention/tracing/globals.py:18`) raises `RuntimeError` if called outside a trace context (`Globals.stack == 0`).

## Pymount mechanism

Controlled by `CONFIG.APP.PYMOUNT` (defaults to `True`):

- On the **first** entry to any tracing context (`Globals.enter`, `globals.py:101`), if not already mounted, the C extension `nnsight._c.py_mount.mount` installs `Object.save` as a method on the base `object` class. This is a one-time global side effect — it is not unmounted on exit.
- Once mounted, every Python object gains `.save()` for the lifetime of the process.
- Set `CONFIG.APP.PYMOUNT = False` to skip pymount entirely. You must then use `nnsight.save()` exclusively.

## Saving non-tensor values

`obj.save()` only works if pymount is enabled (or the object is the `Object`/`torch.Tensor` proxy returned from an eproperty). Always prefer `nnsight.save(...)` for plain Python types:

```python
import nnsight

with model.scan("Hello"):
    dim = nnsight.save(model.transformer.h[0].output.shape[-1])
    paths = nnsight.save([m.path for m in model.modules()])

print(dim, len(paths))
```

## Saving inside generation

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    final = tracer.result.save()
    per_step_logits = list().save()    # save the list, append to it
    for step in tracer.iter[:]:
        per_step_logits.append(model.lm_head.output[0, -1].argmax(dim=-1))
```

For containers built and populated in-place, save the container — its mutated contents persist with it.

## Saving inside `tensor.backward()`

The backward context is a separate interleaving session. Save gradients there:

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    logits = model.lm_head.output

    with logits.sum().backward():
        grad = hs.grad.save()
```

## Remote traces

`.save()` is what tells the remote backend which values to ship back to the client. Without it, the value is computed on the server and discarded.

```python
with model.trace("Hello", remote=True):
    out = model.lm_head.output.detach().cpu().save()
```

Move tensors to CPU before saving for smaller transfers.

## Gotchas

- Forgetting `.save()` is the most common nnsight footgun — your variable will be a stale reference or undefined after `__exit__`.
- `obj.save()` calling user-defined `.save` on a non-nnsight object: prefer `nnsight.save(obj)`.
- `nnsight.save()` is safe to call on the same value multiple times — the saves set is keyed by `id()`.
- Mutating a saved tensor in-place after the trace exits affects whatever the tensor still aliases. Clone if you want isolation: `nnsight.save(x.clone())`.
- Saving inside `model.scan(...)` is required even though the scan exits in the same scope — scan is a tracing context and `Globals.stack` is incremented while inside.

## Related

- `docs/usage/trace.md`
- `docs/usage/scan.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/session.md`

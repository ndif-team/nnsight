---
title: Conditionals and Loops
one_liner: Standard Python `if`/`for` work inside trace contexts because the worker thread sees real tensors.
tags: [usage, control-flow, python]
related: [docs/usage/trace.md, docs/usage/session.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracing/globals.py, src/nnsight/intervention/tracing/base.py:47, src/nnsight/intervention/interleaver.py:264]
---

# Conditionals and Loops

## What this is for

Inside any nnsight tracing context (`model.trace`, `model.generate`, `model.scan`, `model.session`), the body is captured and run in a worker thread. When the worker reads `module.output`, it blocks until the model produces the value via a hook — **and then receives the actual `torch.Tensor`**. So normal Python control flow over those values just works: `if`, `for`, `while`, list/dict comprehensions, function calls, etc.

This replaces the v0.4 proxy-based `nnsight.cond` / `session.iter` machinery — those are gone in v0.5+.

## When to use / when not to use

- Use `if`/`for` freely. There is no special control-flow API to learn.
- Don't use `for` to loop over generation steps — use `tracer.iter[:]` for that. See `docs/usage/iter-all-next.md`.
- Don't use `for` to access modules out of order within one invoke — modules in one invoke must be accessed in forward-pass order. Use multiple invokes instead. See `docs/usage/invoke-and-batching.md`.

## Canonical pattern

```python
with model.trace("Hello"):
    output = model.transformer.h[0].output[0]

    # Real tensor — torch.all returns a real bool
    if torch.all(output < 1e5):
        model.transformer.h[-1].output[0][:] = 0

    final = model.transformer.h[-1].output[0].save()
```

## Looping over prompts (use `model.session`)

```python
prompts = ["Hello", "World", "Test"]

with model.session():
    results = list().save()
    for p in prompts:
        with model.trace(p):
            results.append(model.lm_head.output.argmax(dim=-1))
```

The outer `session()` is what makes this efficient — the trace body is captured once and run in the session frame, so the loop over prompts runs as plain Python.

For remote workloads always wrap in `model.session(remote=True)` so it's one network round-trip for all prompts.

## Branching on tensor values

```python
with model.trace("Hello"):
    hs = model.transformer.h[5].output[0]
    norm = hs.norm(dim=-1).mean()

    if norm > 10.0:
        # Standard Python — no nnsight magic
        model.transformer.h[6].output[0][:] = 0

    final = model.lm_head.output.save()
```

`norm` is a real `torch.Tensor` scalar. `norm > 10.0` is a real bool tensor; Python's `if` triggers `__bool__` on it (works for 0-d tensors).

## Loops inside an invoke

You can write loops as long as they respect the forward-pass-order rule:

```python
with model.trace("Hello"):
    activations = list().save()

    # OK: layers[0]..layers[N-1] are accessed in execution order
    for i in range(len(model.transformer.h)):
        activations.append(model.transformer.h[i].output[0])
```

Reading `model.transformer.h[5].output` then `model.transformer.h[2].output` in the same invoke is a deadlock → `OutOfOrderError`. Loops are fine as long as the iteration order matches the forward pass.

## Why this works

The Tracer captures your with-block source via AST, wraps it in a function definition, and runs it in a worker thread (`src/nnsight/intervention/tracing/base.py:47`). When the function reads an `eproperty` (`.output`, `.input`, ...), the `eproperty.__get__` method blocks on `interleaver.current.request(requester)` (`src/nnsight/intervention/interleaver.py:264`) and returns the real model value once the corresponding hook fires. From that point on, every operation on that value is plain PyTorch.

There is no proxy class wrapping `.output` — the worker just receives the actual tensor.

## Comprehensions and helper functions

```python
with model.trace("Hello"):
    # Real tensors -> real comprehension
    means = [model.transformer.h[i].output[0].mean() for i in range(12)]
    means = nnsight.save(means)

# All entries are real torch tensors, fully resolved.
```

Calling helper functions also works:

```python
def normalize(x):
    return (x - x.mean()) / x.std()

with model.trace("Hello"):
    out = model.transformer.h[0].output[0]
    norm = normalize(out).save()
```

## Gotchas

- `if`/`for` blocks are real Python — they execute in the worker thread. They do **not** create per-step branches in the model's forward pass; they just make decisions about the current activation values.
- `for step in tracer.iter[:]:` is different — that loops over **generation steps**, not Python iterations. See `docs/usage/iter-all-next.md`.
- Python loops cannot reorder module access. The iteration order in your loop must match forward-pass order.
- Inside a `model.session()` body but **outside** an inner trace, you are running plain Python — `module.output` is not accessible there (no interleaver active). Open a `model.trace(...)` first.
- Tensor `__bool__` only returns a scalar bool for 0-d (or single-element) tensors. `if some_tensor:` on a multi-d tensor raises a `RuntimeError` exactly like in vanilla PyTorch — that's not nnsight's fault.

## Related

- `docs/usage/trace.md`
- `docs/usage/session.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/iter-all-next.md`

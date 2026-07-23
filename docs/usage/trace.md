---
title: Trace
one_liner: Single forward pass with interventions via `model.trace(input)`.
tags: [usage, tracing, core]
related: [docs/usage/invoke-and-batching.md, docs/usage/generate.md, docs/usage/pipe.md, docs/usage/scan.md, docs/usage/save.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/envoy.py, src/nnsight/tracing/tracer.py]
---

# Trace

## What this is for

`model.trace(...)` opens an `InterleavingTracer` context that runs a single forward pass of the wrapped model while letting your code read and modify intermediate activations. The body of the `with` block is captured (its source is parsed and compiled — see `src/nnsight/tracing/tracer.py`), then run in a **greenlet worker** (a `Mediator`) that synchronizes with the model's forward pass through hook events.

## When to use / when not to use

- Use for a single forward call (no token-by-token generation).
- Use `model.generate(...)` for multi-token autoregressive output (returns token ids). See `docs/usage/generate.md`.
- Use `model.pipe(...)` to run the whole task pipeline and get its decoded records. See `docs/usage/pipe.md`.
- Use `model.scan(...)` to validate shapes/operations without real compute. See `docs/usage/scan.md`.
- Use `model.edit(...)` for persistent interventions. See `docs/usage/edit.md`.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output.save()
    model.transformer.h[0].output[:] = 0        # zero the first block's output
    logits = model.output.logits.save()

print(hidden.shape, logits.shape)
# torch.Size([1, 10, 768]) torch.Size([1, 10, 50257])
```

## Two equivalent forms

`.trace(input)` with a positional input creates an implicit invoke; `.trace()` without arguments requires explicit `tracer.invoke(...)`:

```python
# Implicit single invoke (input goes to .trace)
with model.trace("Hello"):
    out = model.output.logits.save()

# Explicit invoke (no input on .trace)
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out = model.output.logits.save()
```

`model.trace()` with no input and no invoke block raises:

```
ValueError: trace() needs an input, or at least one `with tracer.invoke(...)` block
```

## Multiple invokes (batched)

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        out_a = model.transformer.h[0].output[0].save()
    with tracer.invoke("The Colosseum is in"):
        out_b = model.transformer.h[0].output[0].save()
```

Each invoke's body sees only its own rows of the batch. See `docs/usage/invoke-and-batching.md` for empty invokes, batching constraints, and barriers.

## Remote execution

`RemoteableMixin.trace` adds `remote=` and `blocking=` kwargs:

```python
with model.trace("Hello", remote=True):
    out = model.output.logits.detach().cpu().save()

# Non-blocking submission
with model.trace("Hello", remote=True, blocking=False) as tracer:
    out = model.output.logits.save()
# tracer.backend.job_id, tracer.backend.job_status
```

`remote='local'` runs against `LocalSimulationBackend` for offline debugging of the remote serialize/deserialize path.

## Input formats

`TransformersModel.trace` accepts every input a forward takes: a string, a list of strings (batched), token-id lists, a 1-D or 2-D tensor, a `BatchEncoding` (positional or unpacked with `**enc`), or `input_ids=`/`attention_mask=` keywords. Mixed formats and unequal lengths are left-padded together into one forward.

## Tracer object

`with model.trace(...) as tracer:` exposes the `InterleavingTracer`. Useful members:

| Member | Purpose |
|---|---|
| `tracer.invoke(*args, **kwargs)` | Add another invoke to the batch |
| `tracer.barrier(n)` | Synchronize across invokes |
| `tracer.cache(...)` | Bulk activation cache (returns a `CacheView`) |
| `tracer.iter[...]` | Per-occurrence targeting (generation steps) |
| `tracer.all()` | Shorthand for `tracer.iter[:]` |
| `tracer.stop()` | Early-exit the current forward pass |
| `tracer.result` | The value the traced call returned |

## Lifecycle

1. `__enter__` parses the with-block source via AST and compiles the body (memoized per site).
2. The body never runs in-place — an `ExitTracingException` skips it.
3. `__exit__` invokes the backend (in-place by default, `RemoteBackend` for `remote=True`).
4. `InterleavingTracer.execute` collects invokes/batched args, then `Envoy.interleave` runs the forward alongside the worker greenlets.
5. Saved values are pushed back into the caller's frame with save-gating (`push_result`).

## Gotchas

- Inside one invoke, modules **must** be accessed in forward-pass order — the worker greenlet blocks on a hook event for each request. Out-of-order access raises `OutOfOrderError`. See `docs/gotchas/out-of-order.md`.
- Values you want after the block **must** be marked with `.save()` / `nnsight.save(...)`. See `docs/usage/save.md`.
- `with model.trace():` with no input and no invoke is a `ValueError`.
- Standard Python `if` / `for` works inside the body — the greenlet worker sees real tensors. See `docs/usage/conditionals-and-loops.md`.
- The traced body must start on its own line (not on the `with` line); combining context managers on one line — `with torch.no_grad(), model.trace(...):` — is fine.
- Tracebacks from inside the trace are cleaned to point at your source lines.

## Related

- `docs/usage/invoke-and-batching.md`
- `docs/usage/generate.md`
- `docs/usage/pipe.md`
- `docs/usage/scan.md`
- `docs/usage/save.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/session.md`

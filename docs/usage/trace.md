---
title: Trace
one_liner: Single forward pass with interventions via `model.trace(input)`.
tags: [usage, tracing, core]
related: [docs/usage/invoke-and-batching.md, docs/usage/generate.md, docs/usage/scan.md, docs/usage/save.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracing/tracer.py:269, src/nnsight/intervention/envoy.py:248, src/nnsight/modeling/mixins/remoteable.py:31, src/nnsight/intervention/tracing/base.py:47]
---

# Trace

## What this is for

`model.trace(...)` opens an `InterleavingTracer` context that runs a single forward pass of the wrapped model while letting your code read and modify intermediate activations. The body of the `with` block is captured, compiled into a function, and executed in a worker thread that synchronizes with the model's forward pass through hook events.

## When to use / when not to use

- Use for a single forward call (no token-by-token generation).
- Use `model.generate(...)` when you need multi-token autoregressive output. See `docs/usage/generate.md`.
- Use `model.scan(...)` when you only need to validate shapes/operations. See `docs/usage/scan.md`.
- Use `model.edit(...)` to make persistent interventions. See `docs/usage/edit.md`.

## Canonical pattern

```python
with model.trace("Hello world"):
    hidden = model.transformer.h[-1].output.save()
    model.transformer.h[0].output[:] = 0
    logits = model.lm_head.output.save()
```

## Two equivalent forms

`.trace(input)` with a positional input creates an implicit invoke; `.trace()` without arguments requires explicit `tracer.invoke(...)`:

```python
# Implicit single invoke (input goes to .trace)
with model.trace("Hello"):
    out = model.lm_head.output.save()

# Explicit invoke (no input on .trace)
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out = model.lm_head.output.save()
```

`model.trace()` with no input and no invoke raises a `ValueError` ("The model did not execute") — see `src/nnsight/intervention/tracing/tracer.py:351`.

## Multiple invokes (batched)

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        out_a = model.lm_head.output[:, -1].save()
    with tracer.invoke("The Colosseum is in"):
        out_b = model.lm_head.output[:, -1].save()
```

See `docs/usage/invoke-and-batching.md` for empty invokes, batching constraints, and barriers.

## Remote execution

`RemoteableMixin.trace` adds `remote=` and `blocking=` kwargs (`src/nnsight/modeling/mixins/remoteable.py:31`):

```python
with model.trace("Hello", remote=True):
    out = model.lm_head.output.save()

# Non-blocking submission
with model.trace("Hello", remote=True, blocking=False) as tracer:
    out = model.lm_head.output.save()
# tracer.backend.job_id, tracer.backend.job_status
```

`remote='local'` runs against `LocalSimulationBackend` for client-side debugging of remote serialization paths.

## Shape inspection

For shape-dependent validation (slicing, reshapes, intervention indexing), use the dedicated `model.scan(input)` context, not `.trace()`. See `docs/usage/scan.md`.

## Tracer object

`with model.trace(...) as tracer:` exposes the `InterleavingTracer`. Useful members:

| Member | Purpose | Source |
|---|---|---|
| `tracer.invoke(*args, **kwargs)` | Add another invoke to the batch | `tracing/tracer.py:433` |
| `tracer.barrier(n)` | Synchronization across invokes | `tracing/tracer.py:551` |
| `tracer.cache(...)` | Activation cache (returns `Cache.CacheDict`) | `tracing/tracer.py:465` |
| `tracer.iter[...]` | Step iterator (generation) | `tracing/tracer.py:453` |
| `tracer.stop()` | Early-exit current forward pass | `tracing/tracer.py:447` |
| `tracer.result` | Forward return value (eproperty) | `tracing/tracer.py:581` |
| `tracer.next(step=1)` | Manually advance the iteration cursor | `tracing/tracer.py:460` |

## Lifecycle

1. `__enter__` → `Tracer.capture()` parses the with-block source via AST and caches it (`tracing/base.py:204`).
2. The body never runs in-place — Python tracing raises `ExitTracingException` to skip it (`tracing/base.py:620`).
3. `__exit__` invokes the configured backend (`ExecutionBackend` by default, `RemoteBackend` for `remote=True`).
4. `InterleavingTracer.execute` calls `_setup_interleaver` (compiles + runs intervention code, collects batched args) then `model.interleave(fn, ...)` (`tracing/tracer.py:412`).
5. Saved values are pulled back into the outer frame via `Tracer.push()` (`tracing/base.py:497`).

## Gotchas

- Inside one invoke, modules **must** be accessed in forward-pass order — the worker thread blocks on a hook event for each request. Out-of-order access raises `OutOfOrderError`. See `docs/gotchas/out-of-order.md`.
- Saved values must be marked with `.save()` or `nnsight.save(...)` to survive past `__exit__`. See `docs/usage/save.md`.
- `with model.trace():` (no input, no invoke) is a `ValueError`. Always provide input via `.trace(...)` or `tracer.invoke(...)`.
- Standard Python `if` / `for` works inside the body — the worker thread sees real tensors. See `docs/usage/conditionals-and-loops.md`.
- Tracebacks from inside the trace are reconstructed by `ExceptionWrapper` to point at your source lines (`tracing/util.py:94`).

## Related

- `docs/usage/invoke-and-batching.md`
- `docs/usage/generate.md`
- `docs/usage/scan.md`
- `docs/usage/save.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/session.md`

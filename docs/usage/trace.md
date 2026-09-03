---
title: Trace
one_liner: Single forward pass with interventions via `model.trace(input)`.
tags: [usage, tracing, core]
related: [docs/usage/invoke-and-batching.md, docs/usage/generate.md, docs/usage/pipe.md, docs/usage/scan.md, docs/usage/save.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/envoy.py, src/nnsight/tracing/tracer.py]
---

# Trace

## What this is for

`model.trace(...)` opens an `InterleavingTracer` context that runs a single forward pass of the wrapped model while letting your code read and modify intermediate activations. The body of the `with` block is captured (its source is parsed and compiled — see `src/nnsight/tracing/tracer.py`), then run in a **greenlet worker** (a `Mediator`) that takes turns with the forward pass: the worker parks on each value it asks for, and the module's controller hands that value over when the model reaches it.

`trace` is for a single forward call. For token-by-token generation reach for `model.generate(...)`, for a whole task pipeline `model.pipe(...)`, for shapes without real compute `model.scan(...)`, and for an intervention that outlives one run `model.edit(...)`.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of"):
    model.transformer.h[0].output[:] = 0             # zero the first block
    hidden = model.transformer.h[-1].output.save()   # then read a later one
    logits = model.output.logits.save()

assert tuple(hidden.shape) == (1, 10, 768)
assert tuple(logits.shape) == (1, 10, 50257)
```

The write comes before the read because block 0 runs before block 11. Every access in one
invoke has to follow the model's own order; see [Gotchas](#gotchas).

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

If that body reads an envoy, you get a different message instead:

```
ValueError: Cannot access `model.transformer.h.0.output` outside of interleaving
```

which reads like an accusation of being outside a trace when you are plainly inside one. The
body of an input-less `trace()` is run once, immediately, to collect the `tracer.invoke(...)`
blocks in it — and that collection pass really does happen before any forward starts. Either
message means the same thing: give `trace()` an input, or open an invoke.

## Multiple invokes (batched)

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        out_a = model.transformer.h[0].output.save()
    with tracer.invoke("The Colosseum is in"):
        out_b = model.transformer.h[0].output.save()
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

`tracer.result` is served after the forward returns, so it is the last thing a block can ask
for. Read `model.output` first if you want both — reversed, the `model.output` request comes
after the model has already produced it:

```python
with model.trace("Hello") as tracer:
    out = model.output.save()
    result = tracer.result.save()

assert result is out
```

## Lifecycle

1. `__enter__` parses the with-block source via AST and compiles the body (memoized per site).
2. The body never runs in-place — an `ExitTracingException` skips it.
3. `__exit__` invokes the backend (in-place by default, `RemoteBackend` for `remote=True`).
4. `InterleavingTracer.execute` collects invokes/batched args, then `Envoy.interleave` runs the forward alongside the worker greenlets.
5. Saved values are pushed back into the caller's frame with save-gating (`push_result`).

## Gotchas

- Inside one invoke, modules **must** be accessed in forward-pass order — the worker parks on each request until the model reaches it. Out-of-order access raises `OutOfOrderError`. See `docs/gotchas/order-and-deadlocks.md`.
- Values you want after the block **must** be marked with `.save()` / `nnsight.save(...)`. See `docs/usage/save.md`.
- `with model.trace():` with no input and no invoke is a `ValueError`.
- Standard Python `if` / `for` works inside the body — the greenlet worker sees real tensors. See `docs/usage/conditionals-and-loops.md`.
- The traced body must start on its own line (not on the `with` line); combining context managers on one line — `with torch.no_grad(), model.trace(...):` — is fine.
- Tracebacks are cleaned of nnsight's own frames, so an `OutOfOrderError` or a missed barrier comes back as three frames ending on the line that waited. An ordinary error raised inside the body still carries the torch and transformers frames under it — your line is there, at the bottom of the pile.

## Related

- `docs/usage/invoke-and-batching.md`
- `docs/usage/generate.md`
- `docs/usage/pipe.md`
- `docs/usage/scan.md`
- `docs/usage/save.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/session.md`

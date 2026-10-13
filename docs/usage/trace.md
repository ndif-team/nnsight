---
title: Trace
one_liner: Single forward pass with interventions via `model.trace(input)`.
tags: [usage, tracing, core]
related: [docs/usage/invoke-and-batching.md, docs/usage/generate.md, docs/usage/pipe.md, docs/usage/scan.md, docs/usage/save.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/envoy.py, src/nnsight/tracing/tracer.py]
---

# Trace

`model.trace(...)` runs a single forward pass of the wrapped model while your code reads and modifies intermediate activations. For token-by-token generation use [generate](generate.md), for a whole task pipeline [pipe](pipe.md), for shapes without real compute [scan](scan.md), and for an intervention that outlives one run [edit](edit.md).

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
invoke has to follow the model's own order, or it raises `OutOfOrderError` ([out-of-order-error](../errors/out-of-order-error.md)).

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

## Multiple invokes and remote

Several prompts in one forward, each with its own block, go through `tracer.invoke(...)`: see [invoke-and-batching](invoke-and-batching.md). `remote=True` (and `blocking=False`) runs the trace on NDIF: see [remote-trace](../remote/remote-trace.md).

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

## Gotchas

- Values you want after the block **must** be marked with `.save()` / `nnsight.save(...)`; see [save](save.md). Plain `if` / `for` work inside the body; see [conditionals-and-loops](conditionals-and-loops.md).
- The traced body must start on its own line (not on the `with` line); combining context managers on one line — `with torch.no_grad(), model.trace(...):` — is fine.
- Tracebacks are cleaned of nnsight's own frames, so an `OutOfOrderError` or a missed barrier comes back as three frames ending on the line that waited. An ordinary error raised inside the body still carries the torch and transformers frames under it — your line is there, at the bottom of the pile.

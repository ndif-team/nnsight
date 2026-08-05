---
title: OutOfOrderError
one_liner: "OutOfOrderError: '<location>' was requested but the model already ran past it — a module value was asked for out of forward-pass order within one block."
tags: [error, execution-order, interleaving]
related: [docs/errors/value-was-not-provided.md, docs/errors/cannot-access-outside-interleaving.md, docs/concepts/threading-and-mediators.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/interleaver.py:83, src/nnsight/intervention/interleaver.py:638, src/nnsight/intervention/interleaver.py:652]
---

# OutOfOrderError

## Symptom

```
nnsight.intervention.interleaver.OutOfOrderError: 'model.transformer.h.1.output.i0' was requested but the model already ran past it
```

Import it from `nnsight.intervention.interleaver` if you want to catch it:

```python
from nnsight.intervention.interleaver import OutOfOrderError
```

## Cause

Each block of intervention code runs in its own **greenlet worker** (a
`Mediator`) that runs in lockstep with the model's forward pass. When the block
reads `module.output`, the worker *parks* until the model reaches that module,
then resumes with the value. A worker can only be served locations **in the order
the model reaches them** — it holds one pending request at a time.

If you ask for layer 1's output *after* layer 5's, layer 1 has already fired and
its value is gone by the time your request arrives. The run finishes with the
worker still parked on `model.transformer.h.1.output`, and
`Interleaver.check_dangling_mediators` (`src/nnsight/intervention/interleaver.py:638`)
throws `OutOfOrderError` into the worker so the traceback points at the exact line
that was waiting.

The `.i0` suffix on the location is the occurrence tag — which visit of that
location the request targets. Without `tracer.iter`, it is always `.i0`; in a
generation loop it counts `.i0`, `.i1`, `.i2`, … per step.

> This is the same class raised by the "model finished, a worker is still waiting"
> case in [value-was-not-provided.md](value-was-not-provided.md). There is no
> separate `MissedProviderError` in this rewrite.

## Common triggers

- Reading modules in reverse order inside one block (`h[5].output` before `h[1].output`).
- Reading the same module's `.output` twice in one block after it has fired.
- Reading a `.grad` for an early layer before a later one inside `with tensor.backward():` — gradients flow in reverse, so access order reverses too (see [docs/usage/backward-and-grad.md](../usage/backward-and-grad.md)).

## Fix

```python
# WRONG — layer 5 fires before layer 1, so the request for h[1] arrives too late
with model.trace("The Eiffel Tower is in"):
    out5 = model.transformer.h[5].output.save()
    out1 = model.transformer.h[1].output.save()   # OutOfOrderError
```

```python
# FIXED — access modules in forward-pass order
with model.trace("The Eiffel Tower is in"):
    out1 = model.transformer.h[1].output.save()
    out5 = model.transformer.h[5].output.save()
```

To genuinely read modules out of forward order, run a second pass with an extra
empty invoke — each invoke is its own worker, so their access orders are
independent:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        out5 = model.transformer.h[5].output.save()
    with tracer.invoke():           # empty invoke = another pass over the same batch
        out1 = model.transformer.h[1].output.save()
```

## Mitigation

- Lay intervention code out top-to-bottom in the order modules run in `print(model)`.
- For backward passes, mirror forward order in reverse inside `with tensor.backward():`.
- Split interleaving access patterns across multiple invokes.

## Another cause: something removed nnsight's hooks

`OutOfOrderError` also fires when the location was never served at all, because
nnsight's forward hooks are gone from the module tree. nnsight installs a
pass-through pre-hook and hook on every module and expects them to stay; another
library that calls `remove()` broadly — some intervention frameworks clear
*every* hook on teardown, not just their own — silently strips them, and the next
trace reports a location the model "already ran past".

The tell is that `.input` breaks while other things still work, and that it
started after running code from another hooking library in the same process.
Re-instrument by walking the tree:

```python
for envoy in [model, *model.modules()]:
    envoy.interleaver.instrument(envoy)
```

## Related

- [value-was-not-provided.md](value-was-not-provided.md) — same class, the "module never fired / iter outran the model" flavor.
- [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) — accessing a value with no trace running at all.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — how greenlet workers park and resume.

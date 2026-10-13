---
title: OutOfOrderError
one_liner: "OutOfOrderError: '<location>' was requested but the model already ran past it — a module value was asked for out of forward-pass order within one block."
tags: [error, execution-order, interleaving]
related: [docs/errors/module-never-fired.md, docs/errors/cannot-access-outside-interleaving.md, docs/concepts/threading-and-mediators.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/interleaver.py]
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
`Interleaver.check_dangling_mediators` throws `OutOfOrderError` into the worker so
the traceback points at the exact line that was waiting.

`model.output` is the location that catches people first. It is the *root* envoy —
the very end of the forward pass — so reading it before any layer strands that
layer's request:

```python
# wrong: model.output runs the whole model, so h[8] has already fired
with model.trace(prompt):
    logits = model.output.logits.save()
    hidden = model.model.layers[8].output.save()   # OutOfOrderError

# right: activations first, model.output last
with model.trace(prompt):
    hidden = model.model.layers[8].output.save()
    logits = model.output.logits.save()
```

That ordering is the natural way to *describe* the task ("give me the prediction and
also layer 8"), which is why it is such a common first error.

The `.i0` suffix on the location is the occurrence tag — which visit of that
location the request targets. Without `tracer.iter`, it is always `.i0`; in a
generation loop it counts `.i0`, `.i1`, `.i2`, … per step.

## Common triggers

- Reading modules in reverse order inside one block (`h[5].output` before `h[1].output`).
- Reading a block before something inside it. Forward order is **depth-first**: a
  module runs *inside* the one that contains it, so `h[0].output` then
  `h[0].attn.output` is out of order — the whole block, attention included, already
  ran. Read inward-out: `h[0].attn.c_proj.input`, then `h[0].attn.output`, then
  `h[0].output`. The same holds on any architecture
  (`layers[0].self_attn.o_proj` inside `layers[0].self_attn`).
- Reading a `.grad` for an early layer before a later one inside `with tensor.backward():` — gradients flow in reverse, so access order reverses too (see [docs/usage/backward-and-grad.md](../usage/backward-and-grad.md)).
- Calling a module yourself without `hook=True`, then reading one of its submodules — the call runs with the trace stood down, so nothing is served (see [calling modules directly](../usage/access-and-modify.md#calling-modules-directly-inside-a-trace)).

Reading the *same* location twice in one block is fine: the second read is
answered from the same served value and hands back the identical object.

```python
with model.trace(prompt):
    first = model.transformer.h[2].output.save()
    again = model.transformer.h[2].output.save()    # `first is again`
```

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

## Inside a `tracer.iter` loop

The same rule applies per step, but past step 0 an out-of-order request binds to
the *next* step instead of raising — silently, or with a "never reached" warning
that blames the loop bound. See
[Pitfalls in iteration.md](../usage/iteration.md#an-out-of-order-body-shifts-by-a-step).

## Another cause: something replaced the module's forward

`OutOfOrderError` also fires when the location was never served at all, because
nnsight's controller is no longer the module's `forward`. nnsight installs the
controller as an instance attribute (`module.__dict__["forward"]`) on every module
and expects it to stay; another library that reassigns `module.forward`, or wraps
the module in a way that bypasses its `forward`, silently removes it, and the next
trace reports a location the model "already ran past".

nnsight warns as soon as it notices — the next time an envoy tree is built or
re-pointed over that module (`"<path>: `forward` was replaced after nnsight wrapped
this module"`) — but nothing calls it in between, so the warning can arrive well
after the assignment, or not before the trace that fails.

The tell is that `.input`/`.output` of some modules break while others still work,
and that it started after running code from another instrumentation library in the
same process. Re-instrumenting means clearing nnsight's own state key as well as
the stale `forward` — `install_controller` (`src/nnsight/intervention/source.py`)
short-circuits on that key, so leaving it behind makes the walk a no-op:

```python
from nnsight.intervention.source import STATE     # "__nnsight__"

for envoy in [model, *model.modules()]:
    envoy._module.__dict__.pop(STATE, None)
    envoy._module.__dict__.pop("forward", None)
    envoy.interleaver.instrument(envoy)
```

## Related

- [module-never-fired.md](module-never-fired.md) — same class, the "module never fired / loop outran the run" flavor.
- [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) — accessing a value with no trace running at all.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — how greenlet workers park and resume.

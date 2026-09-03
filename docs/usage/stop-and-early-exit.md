---
title: Stop and Early Exit
one_liner: Cut a forward pass short with `tracer.stop()`; raises EarlyStopException, swallowed by the interleaver.
tags: [usage, control-flow, early-stop]
related: [docs/usage/trace.md, docs/usage/iter-all-next.md, docs/usage/skip.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py]
---

# Stop and Early Exit

## What this is for

`tracer.stop()` aborts the current run at the point the worker is currently
parked. Any module that hasn't executed yet is **never executed** — the block
raises `EarlyStopException`, which the interleaver treats as a clean early exit
(not an error) and swallows.

This is how you "save what you need and bail" without running the rest of the
forward pass.

## When to use / when not to use

- Use to short-circuit once you've collected the activations you need.
- Use to end generation mid-run when a condition is met.
- Don't use as an error path — `stop()` is a successful early exit.
- Don't use to skip a single module — that's `module.skip(value)` ([skip.md](skip.md)).

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello world") as tracer:
    h0 = model.transformer.h[0].output.save()   # save BEFORE stopping
    tracer.stop()

# Layers 1..N never ran. h0 is populated.
print(h0.shape)   # torch.Size([1, 2, 768])
```

## Stop in generation

```python
import nnsight

with model.generate("Hello", max_new_tokens=20, do_sample=False) as tracer:
    picks = nnsight.save([])
    for step in tracer.all():
        picks.append(model.lm_head.output[0, -1].argmax(dim=-1))
        if len(picks) == 3:
            tracer.stop()
# len(picks) == 3 — the run ended after the third step, not the twentieth
```

Any condition works, since `picks[-1]` is a real tensor: an EOS id, a probability
threshold, a token you were waiting for.

`stop()` ends the whole run, not just one step. It also ends the loop, so the
loop's own bound never has to hold — an open `tracer.all()` is the natural form
here.

## The run's result is gone after a stop

A stop cuts the run off before it returns anything, so there is no result to
read. A `tracer.result.save()` after the `stop()` is unreachable like any other
trailing statement, and moving it into a separate empty invoke does not rescue
it:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello world"):
        h = model.transformer.h[0].output.save()
        tracer.stop()
    with tracer.invoke():
        r = tracer.result.save()
# OutOfOrderError: 'result.i0' was requested but the model already ran past it
```

Save the activations you need before stopping; if you also need the model's
output, do not stop.

## Saving `tracer.result` before `stop()` runs the whole pass

`tracer.result` is served when the call returns, so a worker that asks for it
parks until the run is over — and the `stop()` written below it is only reached
once everything has already run:

```python
with model.trace("Hello world") as tracer:
    h = model.transformer.h[0].output.save()
    r = tracer.result.save()      # parks here until the run finishes
    tracer.stop()                 # fires after the fact
# r is a full CausalLMOutputWithCrossAttentions: r.logits.shape == (1, 2, 50257)
```

The code looks like an early exit and reads back like one. The tell is `r`: it
holds logits for every position, which a stopped run could not have produced.

## Save before you stop

Code after `tracer.stop()` in the same block does not run — Python raises at the
call. So `.save()` (or `nnsight.save(...)`) anything you want to keep **before**
calling `stop()`:

```python
with model.trace("Hello world") as tracer:
    h = model.transformer.h[0].output.save()   # <-- save first
    tracer.stop()
```

## Gotchas

- **`stop()` only works inside an active trace.**
- **Code after `stop()` in the same block never runs** — don't rely on trailing
  side effects.
- **Anything depending on a later module won't be populated.** Requesting a module
  the run never reached (because you stopped before it) raises `OutOfOrderError`.
- **`EarlyStopException` is not an error.** Don't wrap the trace in
  `try/except EarlyStopException` expecting to catch user errors — the interleaver
  already swallows it.
- **The run's result is unreachable after a stop** — from after the `stop()`, and from a separate empty invoke, which raises `OutOfOrderError: 'result.i0'`.
- **A `tracer.result.save()` placed *before* `stop()` defeats the stop** — the worker parks on the result until the run finishes.
- **For per-module bypass without aborting the whole forward, use `module.skip(...)`.**

## Related

- [trace.md](trace.md)
- [iter-all-next.md](iter-all-next.md)
- [skip.md](skip.md)
- [save.md](save.md)

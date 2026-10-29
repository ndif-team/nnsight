---
title: Stop and Early Exit
one_liner: Cut a forward pass short with `tracer.stop()`; raises EarlyStopException, swallowed by the interleaver.
tags: [usage, control-flow, early-stop]
related: [docs/usage/trace.md, docs/usage/iteration.md, docs/usage/skip.md, docs/usage/save.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py]
---

# Stop and Early Exit

`tracer.stop()` ends the run at the location the worker is parked on: modules
after it **never execute**. It raises `EarlyStopException`, which the interleaver
treats as a clean exit and swallows — don't wrap the trace in `try/except` for it.
Use it to "save what you need and bail", or to end generation when a condition is
met. To bypass one module without aborting, use [`module.skip(...)`](skip.md).

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello world") as tracer:
    h0 = model.transformer.h[0].output.save()   # save BEFORE stopping
    tracer.stop()

print(h0.shape)   # torch.Size([1, 2, 768]); layers 1..N never ran
```

Code after `stop()` in the same block does not run, so save first. Requesting a
module the run never reached raises `OutOfOrderError`.

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

Any condition works, since `picks[-1]` is a real tensor (an EOS id, a probability
threshold). `stop()` ends the whole run and the loop with it, so the loop's bound
never has to hold; an open `tracer.all()` is the natural form.

## The location you stop at still happens

The module the worker is parked on has already run when the stop is raised: its
value is served, an edit to it lands, and a cache records it.

```python
with model.trace("Hello world") as tracer:
    cache = tracer.cache(modules=[model.transformer.h[5]]).save()
    model.transformer.h[5].output
    tracer.stop()

print(list(cache.keys()))   # ['model.transformer.h.5']
```

## A stop in one invoke ends the whole batch

Invokes are rows of one forward, so a `stop()` in any of them ends the run for
all. The step it fires on finishes — an invoke parked on the same visit is served
in it — so every invoke records that step:

```python
with model.generate(max_new_tokens=8, do_sample=False) as tracer:
    with tracer.invoke("The capital of France is"):
        picks = nnsight.save([])
        for _ in tracer.all():
            picks.append(model.lm_head.output[0, -1].argmax(dim=-1))
            if len(picks) == 3:
                tracer.stop()
    with tracer.invoke("The capital of Spain is"):
        others = nnsight.save([])
        for _ in tracer.all():
            others.append(model.lm_head.output[0, -1].argmax(dim=-1))
# len(picks) == len(others) == 3
```

The second loop warns that it asked for a step the run did not make
([iteration](iteration.md)). A sibling parked on a later location dies with
`OutOfOrderError`. To stop invokes independently, run separate traces.

## A stop and `tracer.result` don't mix

A stop cuts the run off before it returns, so there is no result: a
`tracer.result.save()` after the `stop()` never runs, and one in a separate empty
invoke raises:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello world"):
        h = model.transformer.h[0].output.save()
        tracer.stop()
    with tracer.invoke():
        r = tracer.result.save()
# OutOfOrderError: 'result.i0' was requested but the model already ran past it
```

Placed *before* the `stop()`, it defeats the stop: `tracer.result` is served when
the call returns, so the worker parks there until the whole pass has run, and the
`stop()` fires after the fact. The tell is a result holding logits for every
position:

```python
with model.trace("Hello world") as tracer:
    h = model.transformer.h[0].output.save()
    r = tracer.result.save()      # parks here until the run finishes
    tracer.stop()                 # fires after the fact
# r.logits.shape == (1, 2, 50257)
```

If you need the model's output, do not stop.

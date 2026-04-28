---
title: Multi-Prompt Comparison
one_liner: Run multiple prompts in one trace using `tracer.invoke(...)` and empty invokes for batch-wide ops; use barriers when invokes share modules.
tags: [pattern, interpretability, batching, comparison]
related: [docs/usage/invoke-and-batching.md, docs/usage/barrier.md, docs/patterns/activation-patching.md, docs/patterns/ablation.md]
sources: [src/nnsight/intervention/tracing/tracer.py:433, src/nnsight/intervention/tracing/tracer.py:551]
---

# Multi-Prompt Comparison

## What this is for

Many interpretability experiments are comparisons: clean vs corrupt, in-distribution vs out-of-distribution, baseline vs ablated, prompt with X vs prompt without X. The natural way to run them in nnsight is **multiple invokes inside one `model.trace()`** rather than multiple separate traces.

Why one trace beats many:

- **Single setup cost.** Each `model.trace(...)` pays a fixed ~0.3 ms compile + thread setup. Many comparisons in one trace amortizes this.
- **Shared interventions.** A no-arg ("empty") invoke runs on the *combined batch* across all input invokes - one place to write a single intervention that applies to every prompt.
- **Cross-invoke variable sharing.** Variables captured in one invoke can be used in a later invoke (with a barrier when both touch the same module).
- **One remote round-trip.** With `remote=True`, all invokes ship as one job.

The architectural concept: each `tracer.invoke(...)` is a worker thread. They run **serially in definition order**. See `docs/usage/invoke-and-batching.md`.

## When to use

- Side-by-side baseline vs ablated / patched / steered.
- Mean-difference / contrast-set computations (positive vs negative prompts).
- Sweeps over a small set of prompts that share interventions.
- Any setup where you used to write `for prompt in prompts: with model.trace(prompt): ...`.

## Canonical pattern

Three invokes in one trace: a baseline, an ablated, and an empty-invoke that summarizes both.

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
LAYER = 9

with model.trace() as tracer:
    with tracer.invoke(prompt):
        baseline = model.lm_head.output[:, -1, :].save()

    with tracer.invoke(prompt):
        model.transformer.h[LAYER].mlp.output[:] = 0
        ablated = model.lm_head.output[:, -1, :].save()

print("baseline argmax:", model.tokenizer.decode(baseline.argmax(-1)[0]))
print("ablated  argmax:", model.tokenizer.decode(ablated.argmax(-1)[0]))
```

No barrier is needed - the two invokes do not share any variable.

## Empty invokes for batch-wide operations

`tracer.invoke()` with no arguments is an **empty invoke**: a separate thread that operates on the *full batch* of all preceding input invokes. Two main uses:

1. **Run an intervention or readout across the whole batch in one place.**
2. **Re-access modules in a different order** - within a single invoke you must read modules in forward-pass order, but each empty invoke is a separate thread that gets its own forward pass replay.

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out_a = model.lm_head.output[:, -1, :].save()        # shape [1, vocab]
    with tracer.invoke(["World", "Test"]):
        out_b = model.lm_head.output[:, -1, :].save()        # shape [2, vocab]

    # Empty invoke: sees the full [3, ...] batch.
    with tracer.invoke():
        full = model.lm_head.output[:, -1, :].save()         # shape [3, vocab]
```

Empty invokes trigger **neither** `_prepare_input()` nor `_batch()` — they reuse the existing batched state from the preceding input invokes. So they work on the base `NNsight` (which does not implement batching) too — one input invoke + as many empty invokes as you want.

## When you need a barrier

If two invokes both touch `.output` (or `.input`) of the same module and you want to share a variable between them, you **must** use `tracer.barrier(n)` to synchronize them. Without it, the second invoke runs before the first has materialized the value, and you get a `NameError`.

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)

    # Capture clean residual.
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[:, -1, :]
        barrier()

    # Patch into corrupt run.
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[:, -1, :] = clean_hs
        patched = model.lm_head.output[:, -1, :].save()
```

See `docs/usage/barrier.md` and `docs/patterns/activation-patching.md` for more examples.

## Variations

### Sweep over prompts

```python
prompts = ["The cat sat on the", "A dog ran on the", "The bird flew on the"]

with model.trace() as tracer:
    last_logits = []
    for p in prompts:
        with tracer.invoke(p):
            last_logits.append(model.lm_head.output[:, -1, :].save())

for p, lg in zip(prompts, last_logits):
    print(p, "->", model.tokenizer.decode(lg.argmax(-1)[0]))
```

### Mean / difference of activations

```python
positive = ["I love this", "This is wonderful"]
negative = ["I hate this", "This is awful"]
LAYER = 6

with model.trace() as tracer:
    pos_a, neg_a = [], []
    for p in positive:
        with tracer.invoke(p):
            pos_a.append(model.transformer.h[LAYER].output[:, -1, :].save())
    for p in negative:
        with tracer.invoke(p):
            neg_a.append(model.transformer.h[LAYER].output[:, -1, :].save())

import torch
pos = torch.cat([a for a in pos_a]).mean(0)
neg = torch.cat([a for a in neg_a]).mean(0)
direction = pos - neg
```

### Pre-batched input (no invokes)

If you just want a forward pass on a batch, pass the list directly:

```python
with model.trace(["Hello", "World"]):
    last = model.lm_head.output[:, -1, :].save()    # shape [2, vocab]
```

This is the right choice when you do *not* need different interventions per prompt.

### Comparing clean / corrupt with shared intervention

Use one input invoke per prompt, then an empty invoke that applies a uniform intervention across the batch:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        pass
    with tracer.invoke("The Colosseum is in"):
        pass
    with tracer.invoke():                           # empty: full [2, S]
        model.transformer.h[5].mlp.output[:] = 0    # ablate uniformly
        logits = model.lm_head.output[:, -1, :].save()  # [2, vocab]
```

## Interpretation tips

- **Pre-batching vs invokes** are equivalent for *plain* runs. Use invokes when each prompt needs its own intervention or you want side-by-side baseline/condition logits in one trace.
- **Empty invokes operate on the *combined* batch.** Their interventions affect every prompt simultaneously, which is what you usually want for fair comparisons.
- **Order matters: invokes run serially in definition order.** A variable defined in invoke 1 is visible in invoke 2 (provided it has been materialized - hence barriers when both touch the same module).
- **Save inside the trace, not after.** `model.lm_head.output[:, -1, :].save()` inside an invoke. If you forget `.save()`, the value is gone.

## Gotchas

- Multiple input invokes require the model to implement batching. `LanguageModel` does. Base `NNsight` does not - use one input invoke + empty invokes, or pre-batch the input.
- Inside one invoke, modules must be accessed in forward order. To read modules in arbitrary order, use multiple empty invokes.
- Cross-invoke variables on the *same* module need a barrier. See `docs/usage/barrier.md`.
- A `tracer.barrier(n)` requires exactly `n` calls to `barrier()`. If you have a non-participating extra invoke, do not include it in the count.

## Related

- `docs/usage/invoke-and-batching.md` - The full reference for invokes, empty invokes, and batching.
- `docs/usage/barrier.md`
- [activation-patching](activation-patching.md) - The canonical example of a same-module cross-invoke pattern.
- [ablation](ablation.md), [steering](steering.md)

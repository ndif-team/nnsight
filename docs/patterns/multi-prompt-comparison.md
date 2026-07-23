---
title: Multi-Prompt Comparison
one_liner: Run multiple prompts in one trace using `tracer.invoke(...)` and empty invokes for batch-wide ops; use `tracer.barrier(n)` when invokes share a value.
tags: [pattern, interpretability, batching, comparison]
related: [docs/usage/invoke-and-batching.md, docs/usage/barrier.md, docs/patterns/activation-patching.md, docs/patterns/ablation.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/barrier.py, src/nnsight/intervention/batching.py]
---

# Multi-Prompt Comparison

## What this is for

Many interpretability experiments are comparisons: clean vs corrupt, baseline vs
ablated, prompt-with-X vs prompt-without-X. Run them as **multiple invokes inside
one `model.trace()`** rather than separate traces.

Why one trace beats many:

- **Single setup cost**, amortized across every comparison.
- **Shared interventions.** An empty (`tracer.invoke()`) invoke runs on the
  *combined batch* of all input invokes — one place for an intervention that applies
  to every prompt.
- **Cross-invoke value sharing.** A value read in one invoke can be used in a later
  invoke (with a barrier when both touch the same module).
- **One remote round-trip.** With `remote=True`, all invokes ship as one job.

Each `tracer.invoke(...)` block is a **worker** (a greenlet), and they resume in the
order the model reaches what each asked for. See `docs/usage/invoke-and-batching.md`.

## When to use

- Side-by-side baseline vs ablated / patched / steered.
- Mean-difference / contrast-set computations (positive vs negative prompts).
- Sweeps over a small set of prompts that share interventions.
- Anywhere you used to write `for p in prompts: with model.trace(p): ...`.

## Canonical pattern

Two invokes in one trace: a baseline and an ablated run.

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
LAYER = 9

with model.trace() as tracer:
    with tracer.invoke(prompt):
        baseline = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(prompt):
        model.transformer.h[LAYER].mlp.output[:] = 0
        ablated = model.lm_head.output[:, -1, :].save()

print(model.tokenizer.decode(baseline.argmax(-1)[0]))   # ' Paris'
print(model.tokenizer.decode(ablated.argmax(-1)[0]))    # ' London'
```

No barrier is needed — the two invokes do not share a variable.

## Empty invokes for batch-wide operations

`tracer.invoke()` with no arguments is an **empty invoke**: a worker that sees the
*whole batch* of all preceding input invokes, with no row scoping.

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        a = model.lm_head.output[:, -1, :].save()        # [1, vocab]
    with tracer.invoke(["World", "Test"]):
        b = model.lm_head.output[:, -1, :].save()        # [2, vocab]
    with tracer.invoke():
        full = model.lm_head.output[:, -1, :].save()     # [3, vocab] — whole batch
```

```
a.shape (1, 50257)   b.shape (2, 50257)   full.shape (3, 50257)
```

An empty invoke reuses the batched state from the preceding input invokes — no extra
input prep or batching — so it works on the base `NNsight` too (one input invoke +
as many empty invokes as you like).

## When you need a barrier

If one invoke hands a value to another and both touch the same module, use
`tracer.barrier(n)` so the reader waits for the writer. `barrier(n)` returns a
callable; every participating block calls it, and the last to arrive releases them
all — everything above the barrier has happened before anything below it.

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("Madison Square Garden is in"):
        embeds = model.transformer.wte.output          # read source embeddings
        barrier()
        source_tok = model.lm_head.output[:, -1, :].argmax(-1).save()
    with tracer.invoke("_ _ _ _ _"):
        barrier()
        model.transformer.wte.output = embeds          # ...swap them into this run
        recv_tok = model.lm_head.output[:, -1, :].argmax(-1).save()
```

The receiver, fed the source's embeddings, reproduces the source's prediction.
`barrier(n)` must be reached by exactly `n` blocks — fewer and it never releases (the
blocks left waiting report it when the run ends rather than hanging). See
`docs/usage/barrier.md` and [activation-patching](activation-patching.md).

## Variations

### Sweep over prompts

```python
import nnsight

prompts = ["The cat sat on the", "A dog ran on the", "The bird flew on the"]
with model.trace() as tracer:
    outs = nnsight.save([])
    for p in prompts:
        with tracer.invoke(p):
            outs.append(model.lm_head.output[:, -1, :])

for p, lg in zip(prompts, outs):
    print(p, "->", repr(model.tokenizer.decode(lg.argmax(-1)[0])))
```

```
The cat sat on the -> ' floor'
A dog ran on the -> ' ground'
The bird flew on the -> ' ground'
```

### Mean / difference of activations (a steering direction)

```python
import torch
import nnsight
positive = ["I love this", "This is wonderful"]
negative = ["I hate this", "This is awful"]
LAYER = 6

with model.trace() as tracer:
    pos_a = nnsight.save([])
    neg_a = nnsight.save([])
    for p in positive:
        with tracer.invoke(p):
            pos_a.append(model.transformer.h[LAYER].output[:, -1, :])
    for p in negative:
        with tracer.invoke(p):
            neg_a.append(model.transformer.h[LAYER].output[:, -1, :])

direction = torch.cat(pos_a).mean(0) - torch.cat(neg_a).mean(0)   # [768]
```

### Pre-batched input (no invokes)

If you just want a forward on a batch and every row gets the same treatment, pass
the list directly:

```python
with model.trace(["Hello", "World"]):
    last = model.lm_head.output[:, -1, :].save()    # [2, vocab]
```

Use invokes instead when each prompt needs its own intervention.

### Clean / corrupt with one shared intervention

One input invoke per prompt, then an empty invoke applying a uniform intervention
across the batch:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        pass
    with tracer.invoke("The Colosseum is in"):
        pass
    with tracer.invoke():                              # empty: whole [2, S] batch
        model.transformer.h[5].mlp.output[:] = 0       # ablate uniformly
        logits = model.lm_head.output[:, -1, :].save() # [2, vocab]
```

## Interpretation tips

- **Pre-batching vs invokes** are equivalent for *plain* runs. Use invokes when each
  prompt needs its own intervention or you want side-by-side logits in one trace.
- **Empty invokes hit the whole batch** — their interventions affect every prompt at
  once, which is what you want for fair comparisons.
- **Save inside the trace.** A value without `.save()` is gone once the block exits.

## Gotchas

- **Multiple input invokes need a batching model.** `TransformersModel` batches; the
  base `NNsight` does not — use one input invoke + empty invokes, or pre-batch.
- **Read modules in forward order within one invoke.** To read in a different order,
  use another (empty) invoke — a separate worker gets its own pass.
- **Cross-invoke values on the *same* module need a barrier**, and a read must park
  *past* the write's location first (reading a not-yet-bound name raises
  `NameError`). See `docs/usage/barrier.md`.
- **`tracer.barrier(n)` needs exactly `n` callers** — do not count a
  non-participating invoke.
- **A GPT-2 block's `.output` is a plain tensor**; overwrite the whole tensor
  (`h[L].output = x`) rather than an in-place slice of a tuple element.

## Related

- `docs/usage/invoke-and-batching.md` — invokes, empty invokes, batching.
- `docs/usage/barrier.md`
- [activation-patching](activation-patching.md) — the canonical same-module cross-invoke pattern.
- [ablation](ablation.md), [steering](steering.md)

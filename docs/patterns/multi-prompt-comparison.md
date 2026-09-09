---
title: Multi-Prompt Comparison
one_liner: Run multiple prompts in one trace using `tracer.invoke(...)`, with empty invokes for batch-wide ops and `tracer.barrier(n)` when one invoke hands a value to another.
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
- **Cross-invoke value sharing.** A value read in one invoke can be used in another,
  with a barrier when the consumer writes it.
- **One remote round-trip.** With `remote=True`, all invokes ship as one job.

Each `tracer.invoke(...)` block is a **worker** (a greenlet), and they resume in the
order the model reaches what each asked for. See `docs/usage/invoke-and-batching.md`.

## When to use

- Side-by-side baseline vs ablated / patched / steered.
- Mean-difference / contrast-set computations (positive vs negative prompts).
- Sweeps over a small set of prompts that share interventions.
- Any sweep otherwise written as `for p in prompts: with model.trace(p): ...`.

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
*whole batch* every input invoke contributes, with no row scoping. The batch is
assembled before any worker starts, so an empty invoke written first sees the rows
of the invokes below it just the same.

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

An empty invoke reuses the batch the input invokes built — no extra input prep, no
call into the model's batching methods — so it works on the base `NNsight` too (one
input invoke plus as many empty invokes as you like).

## When you need a barrier

An invoke can read a name a sibling bound once it has parked at a location the model
reaches after the binding; a consumer that *writes* the value has not parked at all
when it reads it, because the assignment evaluates its right-hand side first. Order
those with `tracer.barrier(n)`. Every participating block calls it, and the last to
arrive releases them all, so everything above the barrier has happened before
anything below it. The full rule is in `docs/usage/invoke-and-batching.md`.

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("Madison Square Garden is in the city of"):
        embeds = model.transformer.wte.output          # read source embeddings
        barrier()
        source_tok = model.lm_head.output[:, -1, :].argmax(-1).save()
    with tracer.invoke("_ _ _ _ _ _ _ _"):
        barrier()
        model.transformer.wte.output = embeds          # ...swap them into this run
        recv_tok = model.lm_head.output[:, -1, :].argmax(-1).save()

# both decode to ' New'
```

The receiver, fed the source's embeddings, reproduces the source's prediction.
`barrier(n)` must be reached by exactly `n` blocks. Fewer and the round never
releases, which surfaces as a `ValueError` when the run ends rather than a hang;
more and it releases early, and the block it let through raises `NameError` on the
value it was waiting for. See `docs/usage/barrier.md` and
[activation-patching](activation-patching.md).

## Variations

### Sweep over prompts

```python
import nnsight

prompts = ["The cat sat on the", "A dog ran on the", "The bird flew on the"]
with model.trace() as tracer:
    outs = nnsight.save({})
    for p in prompts:
        with tracer.invoke(p):
            outs[p] = model.lm_head.output[:, -1, :]

for p in prompts:
    print(p, "->", repr(model.tokenizer.decode(outs[p].argmax(-1)[0])))
```

```
The cat sat on the -> ' floor'
A dog ran on the -> ' ground'
The bird flew on the -> ' ground'
```

Key the results by whatever the loop varies rather than appending to a list. Values
appended from several invokes come back in the order the model reached them, so
zipping the list against `prompts` misattributes every row as soon as the invokes
stop reading the same module.

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

## Gotchas

- **Multiple input invokes need a batching model.** `TransformersModel` batches; the
  base `NNsight` does not — use one input invoke + empty invokes, or pre-batch.
- **Read modules in forward order within one invoke.** To read in a different order,
  use another (empty) invoke — a separate worker gets its own pass.
- **A cross-invoke read must park past the binder first**, and a consumer that
  writes cannot park at all — give that one a barrier. Reading a name the producer
  has not bound yet raises `NameError`. See `docs/usage/barrier.md`.
- **Prompts of different lengths are padded to the batch's longest.** An absolute
  position index therefore means a different token depending on what else is in the
  batch; `[:, -1]` is the one that holds. See
  `docs/usage/invoke-and-batching.md`.
- **`tracer.barrier(n)` needs exactly `n` callers** — do not count a
  non-participating invoke.
- **A GPT-2 block's `.output` is a plain tensor**; overwrite the whole tensor
  (`h[L].output = x`) rather than an in-place slice of a tuple element.

## Related

- `docs/usage/invoke-and-batching.md` — invokes, empty invokes, batching.
- `docs/usage/barrier.md`
- [activation-patching](activation-patching.md) — the canonical same-module cross-invoke pattern.
- [ablation](ablation.md), [steering](steering.md)

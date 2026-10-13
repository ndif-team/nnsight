---
title: Batching and Invokers
one_liner: Each tracer.invoke() is its own worker over its own rows of one combined batch; an empty invoke sees the whole batch; a value is scoped to an invoke only when its leading dimension is the batch size (or a multiple of it).
tags: [concept, mental-model, batching, invokers]
related: [docs/concepts/threading-and-mediators.md, docs/usage/invoke-and-batching.md, docs/developing/batching-internals.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/batching.py]
---

# Batching and Invokers

## One forward, several workers

`tracer.invoke(...)` lets several inputs share **one** forward pass while each runs
its own intervention code. Every invoke is its own worker
([threading-and-mediators.md](threading-and-mediators.md)) with a slice of the
combined batch — its **rows** — and sees only those rows of each activation:

```python
from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace() as tracer:
    with tracer.invoke("Hello"):
        a = model.lm_head.output[:, -1].save()        # (1, 50257) — row 0
    with tracer.invoke(["World", "Test foo bar"]):
        b = model.lm_head.output[:, -1].save()        # (2, 50257) — rows 1-2
    with tracer.invoke():
        whole = model.lm_head.output[:, -1].save()    # (3, 50257) — every row
```

`trace("input")` is the same thing with one implicit invoke.

**The scoping rule:** a value is narrowed to an invoke's rows only when its leading
dimension is the combined batch size, or a whole multiple of it (a flattened
`(batch*seq, hidden)` activation). Anything else — a causal mask, rotary `cos`/`sin`,
a vision tower's patch axis — is served whole to every invoke, and *writing* to it
warns, because the write would apply to every invoke. A coincidental multiple (four
experts against two invokes) is sliced as if it were rows. A replacement for a scoped
value must keep the invoke's row count; one of the wrong height is spliced in as
given and fails later, or not at all.

Merging two or more input invokes is the model class's job: `TransformersModel` does
it; bare `NNsight` raises
[`NotImplementedError`](../errors/batching-not-implemented.md). How rows are
assigned, narrowed and spliced back is in
[developing/batching-internals.md](../developing/batching-internals.md).

## Empty invokes

`tracer.invoke()` with no arguments contributes no rows and sees the **whole**
batch, wherever it is written. It is a separate worker, so its access order is
independent of the other invokes' (inside it, forward order still applies). It never
needs batching, so it works on bare `NNsight`. A trace whose *only* invoke is empty
has no rows to run and fails in the model's input handling — give it an input.

## Skipping in a batch

A skipped module produces no output to slice, so each invoke's replacement is
concatenated into the full-batch output. That only works if **every** invoke skips
the module (or none does):

```
ValueError: A batched `.skip()` has to cover every row: skip the module in every
invoke, or none — a shared forward can't run for only the rows an invoke left unskipped.
```

## Ordering across invokes

Within an invoke, forward order. Across invokes, workers resume in the order the
model reaches what each asked for, not the order they are written; passing a value
from one invoke to another needs the rules in
[usage/invoke-and-batching.md](../usage/invoke-and-batching.md#cross-invoke-value-sharing)
and sometimes a [barrier](../usage/barrier.md).

## Related

- [usage/invoke-and-batching.md](../usage/invoke-and-batching.md) — recipes.
- [developing/batching-internals.md](../developing/batching-internals.md) — the `Batcher`, custom batch layouts.

---
title: Barrier
one_liner: Cross-invoke synchronization point for handing values across invokes that touch the same module.
tags: [usage, batching, synchronization]
related: [docs/usage/invoke-and-batching.md, docs/usage/access-and-modify.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/barrier.py, src/nnsight/intervention/tracer.py, src/nnsight/intervention/interleaver.py]
---

# Barrier

## What this is for

The blocks of a trace — one per `with tracer.invoke(x):` — run in the order the
model reaches what each asked for, not the order they were written. A value one
block reads and another block writes is only correct if the read happened first,
and neither block can see the other's progress.

`tracer.barrier(n)` is that meeting point. Every block that holds the barrier
calls it; each waits, and the last to arrive releases them all. So everything
written **above** a barrier has happened before anything written **below** one.

Reach for it whenever **two (or more) invokes hand a value across the same
module**.

## When to use / when not to use

- Use when a later invoke needs a value an earlier invoke produced from the *same*
  module.
- Don't use when invokes touch entirely different modules — shared invoke scope
  already handles that (see [invoke-and-batching.md](invoke-and-batching.md)).
- Don't use as a substitute for `tracer.stop()` or `module.skip()`.

## Canonical pattern (embedding transfer)

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.pipe(max_new_tokens=3, do_sample=False) as tracer:
    barrier = tracer.barrier(2)          # 2 participating invokes

    with tracer.invoke("Madison Square Garden is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()                        # signal: embeddings are read
        result = tracer.result.save()

    with tracer.invoke("_ _ _ _ _ _ _ _ _"):
        barrier()                        # wait until the source read its embeddings
        model.transformer.wte.output = embeddings
```

The second prompt is only underscores, yet — because it generates from the first
prompt's embeddings — it produces the same continuation.

## Why a barrier is required here

Both invokes touch `transformer.wte.output`. Without a barrier the second invoke's
worker would try to swap in `embeddings` before the first worker had read it —
`NameError`, because the name isn't bound yet. The barrier pins the ordering: the
first invoke parks at its `barrier()` with `embeddings` already read, the second
runs up to *its* `barrier()`, and the last one through releases both.

## More than two participants

`tracer.barrier(n)` supports any `n`. A barrier of three fans one invoke's value
out to two receivers:

```python
receiver = "_ _ _ _ _ _ _ _ _"
with model.pipe(max_new_tokens=3, do_sample=False) as tracer:
    barrier = tracer.barrier(3)
    with tracer.invoke("Madison Square Garden is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()
        result = tracer.result.save()
    with tracer.invoke(receiver):
        barrier()
        model.transformer.wte.output = embeddings
    with tracer.invoke(receiver):
        barrier()
        model.transformer.wte.output = embeddings
```

No block passes the barrier until all three reach it — every "before" happens
before any "after".

## Reusable

A single `Barrier` empties its waiting list on release, so the same object can be
used again — each round waits for its own `n` arrivals:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("A"):
        a5 = model.transformer.h[5].output
        barrier()
        a8 = model.transformer.h[8].output
        barrier()
    with tracer.invoke("B"):
        barrier()
        x = a5          # available after the first barrier
        barrier()
        y = a8          # available after the second
```

For different participant counts at different points, create separate barriers.

## Several sites: one round each, and fence every read

With more than one site in play, the rule is that **no worker may request a
location past site *i* until everyone is done with site *i*** — requesting a
later location is what drives the model forward. So each source's read has to
sit *behind* the rounds for every earlier site, not up front:

```python
with model.trace() as tracer:
    b = tracer.barrier(3)

    with tracer.invoke(source_a):
        a5 = model.transformer.h[5].output[0]   # site 1: mine, read it now
        b()                                     # round 1
        b()                                     # round 2: not mine, still attend

    with tracer.invoke(source_b):
        b()                                     # round 1: not mine — wait first
        a8 = model.transformer.h[8].output[0]   # site 2: only now may I read
        b()                                     # round 2

    with tracer.invoke(base):
        b()
        h5 = model.transformer.h[5].output[0]   # site 1
        h5[:, -1] = a5[:, -1]
        b()
        h8 = model.transformer.h[8].output[0]   # site 2
        h8[:, -1] = a8[:, -1]
        logits = model.lm_head.output[:, -1].save()
```

Every invoke calls the barrier in every round, including rounds for sites it
does not touch — a round only releases once all `n` participants arrive.

**The failure to avoid** is hoisting the reads. If `source_b` reads `h[8]`
*before* the first round, the model is driven past `h[5]` before `base` has
written there, and the write raises `OutOfOrderError`:

```python
    with tracer.invoke(source_a):
        a5 = model.transformer.h[5].output[0]
        b()
    with tracer.invoke(source_b):
        a8 = model.transformer.h[8].output[0]   # too early — advances past h[5]
        b()
```

The base's write does not need its own round. Workers are greenlets and do not
preempt each other: once a round releases, `base` runs its write at site *i* to
completion before it parks on site *i+1*, so the write lands while the model is
still at site *i*.

## Gotchas

- **`n` must equal the number of invokes that call `barrier()`.** If fewer arrive,
  it never releases and the run ends with
  `ValueError: A barrier was never reached by every block it waits for; check the
  count it was created with`.
- **The return value is called, not entered.** `barrier = tracer.barrier(n)` then
  `barrier()` — it is not a context manager.
- **A barrier nobody calls is inert** — creating `tracer.barrier(n)` and never
  calling it is harmless.
- **A barrier is per-trace.** Create it inside the `with model.trace()` block.
- **An early `return` that skips a `barrier()` in one invoke hangs that round** —
  every participant must reach it.
- **Reading a later site too early breaks an earlier one.** With several sites,
  put each source's read *after* the rounds for every site before it; requesting
  a location is what advances the model. See above.

## Related

- [invoke-and-batching.md](invoke-and-batching.md)
- [access-and-modify.md](access-and-modify.md)
- [trace.md](trace.md)

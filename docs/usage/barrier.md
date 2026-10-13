---
title: Barrier
one_liner: Cross-invoke synchronization point for handing a value from one invoke to another.
tags: [usage, batching, synchronization]
related: [docs/usage/invoke-and-batching.md, docs/usage/access-and-modify.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/barrier.py, src/nnsight/intervention/tracer.py, src/nnsight/intervention/interleaver.py]
---

# Barrier

The blocks of a trace — one per `with tracer.invoke(x):` — run in the order the
model reaches what each asked for, not the order they were written. A value one
block reads and another writes is only correct if the read happened first.
`tracer.barrier(n)` is the meeting point: every participating block calls it, each
waits, and the last to arrive releases them all. Everything written **above** a
barrier happens before anything written **below** one.

## When to use

A block can read a name another block bound only once it has parked at a location
the model reaches after the binding (the rule is in
[invoke-and-batching](invoke-and-batching.md#cross-invoke-value-sharing)). Use a
barrier when it cannot get there:

- **The consumer writes.** `module.output[...] = donor` evaluates `donor` before
  the attribute access parks, so the write never buys the consumer a park.
- **The consumer acts at or before the producer's location**, e.g. both at `wte`,
  the first module, where there is nothing earlier to park on.

A read-only consumer that can park past the producer needs no barrier, but use one
anyway whenever the block writes: park-past depends on where two lines sit, and
inserting a line above the read turns it into a `NameError`.

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

The second prompt is only underscores, yet it generates from the first prompt's
embeddings and produces the same continuation. Without the barrier the swap reads
`embeddings` before the donor has run and raises `NameError`.

`n` can be anything: `tracer.barrier(3)` with two receivers fans one value out to
both, and no block passes until all three arrive.

## Reusable

A `Barrier` empties its waiting list on release, so the same object serves several
rounds, each waiting for its own `n` arrivals:

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

**No worker may request a location past site *i* until everyone is done with site
*i*** — requesting a later location is what drives the model forward. So each
source's read sits *behind* the rounds for every earlier site:

```python
with model.trace() as tracer:
    b = tracer.barrier(3)

    with tracer.invoke(source_a):
        a5 = model.transformer.h[5].output   # site 1: mine, read it now
        b()                                     # round 1
        b()                                     # round 2: not mine, still attend

    with tracer.invoke(source_b):
        b()                                     # round 1: not mine — wait first
        a8 = model.transformer.h[8].output   # site 2: only now may I read
        b()                                     # round 2

    with tracer.invoke(base):
        b()
        h5 = model.transformer.h[5].output   # site 1
        h5[:, -1] = a5[:, -1]
        b()
        h8 = model.transformer.h[8].output   # site 2
        h8[:, -1] = a8[:, -1]
        logits = model.lm_head.output[:, -1].save()
```

Every invoke calls the barrier in every round, including rounds for sites it does
not touch. Hoisting `source_b`'s `h[8]` read above round one drives the model past
`h[5]` before `base` can write there; `base` never reaches round one and the run
ends with the never-reached error below. The base's write needs no round of its
own: workers do not preempt each other, so once a round releases, `base` finishes
its write at site *i* before parking on site *i+1*.

## Gotchas

- **`n` must equal the number of blocks that call `barrier()`.** Count too high —
  or a block that skips its `barrier()` on a branch, or never reaches it after an
  earlier mistake — and the run ends with `ValueError: A barrier was never reached
  by every block it waits for; check the count it was created with`.
- **Counting too low is the dangerous direction.** `tracer.barrier(2)` called by
  three blocks releases on the second arrival, before the producer has run, and the
  consumer reports `NameError: name 'donor' is not defined`, pointing nowhere near
  the barrier. If a barriered handoff raises `NameError`, recount the callers.
- **The return value is called, not entered** — `barrier()`, not `with barrier:`.
- **Create the barrier inside the `with model.trace()` block.** A `Barrier(n)`
  constructed by hand can be passed into several traces, but one left holding
  waiters from a trace that raised carries them into its next round.
- **Not available on vLLM**, where each invoke is a separate engine request;
  `tracer.barrier(n)` raises `NotImplementedError`. Use two traces instead — see
  [Passing values between invokes](../models/vllm.md#passing-values-between-invokes).
- **A barrier inside `tracer.iter[...]` synchronizes each step**: every participant
  calls it once per iteration, and the handoff repeats for every generated token.

---
title: Barrier
one_liner: Cross-invoke synchronization point for handing a value from one invoke to another.
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

## When to use

A block can only read a name another block bound once it has parked at a location
the model reaches after the binding — the rule is in
[invoke-and-batching.md](invoke-and-batching.md#cross-invoke-value-sharing). A
barrier is what you use when it cannot get there:

- **The consumer writes.** `module.output[...] = donor` evaluates `donor` before
  the attribute access parks the worker, so the write itself never buys the
  consumer a park, whichever module it writes to.
- **The consumer has to act at or before the producer's location.** The embedding
  transfer below is the extreme case: `wte` is the first module, so there is
  nothing earlier to park on.

A read-only consumer that *can* park past the producer needs no barrier. Reach for
one anyway whenever the block writes: park-past depends on where two lines sit
relative to each other, and inserting a line above the read turns it into a
`NameError`.

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

`wte` is the first module the model reaches, so the receiving invoke has nowhere
earlier to park: its first statement is the swap, and the swap reads `embeddings`
before it parks at all. Without the barrier that read raises `NameError`, because
the donor worker has not run yet. The barrier pins the ordering instead: the first
invoke parks at its `barrier()` with `embeddings` already read, the second runs up
to *its* `barrier()`, and the last one through releases both.

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

Every invoke calls the barrier in every round, including rounds for sites it
does not touch — a round only releases once all `n` participants arrive.

**The failure to avoid** is hoisting the reads. If `source_b` reads `h[8]`
*before* the first round, the model is driven past `h[5]` before `base` can write
there. `base` is then parked on a location the model has passed, never arrives at
round one, and the run ends with `ValueError: A barrier was never reached by every
block it waits for; check the count it was created with`:

```python
    with tracer.invoke(source_a):
        a5 = model.transformer.h[5].output
        b()
    with tracer.invoke(source_b):
        a8 = model.transformer.h[8].output   # too early — advances past h[5]
        b()
```

The base's write does not need its own round. Workers are greenlets and do not
preempt each other: once a round releases, `base` runs its write at site *i* to
completion before it parks on site *i+1*, so the write lands while the model is
still at site *i*.

## Gotchas

- **`n` must equal the number of blocks that call `barrier()`.** Count too high and
  the round never releases; the run does not hang, it ends with `ValueError: A
  barrier was never reached by every block it waits for; check the count it was
  created with`. That is also what you get when a block skips its `barrier()` on a
  branch, or when an earlier mistake stops it from reaching the call at all.
- **Counting too low is the dangerous direction.** `tracer.barrier(2)` called by
  three blocks releases on the second arrival, before the producer has run, and the
  consumer it let through reports `NameError: name 'donor' is not defined` — an
  error that names a variable and points nowhere near the barrier. If a barriered
  handoff raises `NameError`, recount the callers.
- **The return value is called, not entered.** `barrier = tracer.barrier(n)` then
  `barrier()` — it is not a context manager.
- **A barrier nobody calls is inert** — creating `tracer.barrier(n)` and never
  calling it is harmless.
- **Create the barrier inside the `with model.trace()` block.** The name lives in
  the trace body and does not survive it. A `Barrier(n)` constructed by hand can be
  passed into several traces, but one left holding waiters from a trace that raised
  carries them into its next round.
- **Not available on vLLM.** Each invoke there is a separate engine request,
  scheduled independently, so the blocks never run against one forward and a
  barrier could not release; `tracer.barrier(n)` raises `NotImplementedError`.
  Hand values across with two traces instead, where a saved value ships with the
  next block — see
  [Passing values between invokes](../models/vllm.md#passing-values-between-invokes).
- **Reading a later site too early breaks an earlier one.** With several sites,
  put each source's read *after* the rounds for every site before it; requesting
  a location is what advances the model. See above.
- **A barrier inside `tracer.iter[...]` synchronizes each step**, not the whole
  generation: every participant calls it once per iteration, and the handoff
  repeats for every generated token.

## Related

- [invoke-and-batching.md](invoke-and-batching.md) — the cross-invoke rule this
  page synchronizes.
- [access-and-modify.md](access-and-modify.md)
- [trace.md](trace.md)

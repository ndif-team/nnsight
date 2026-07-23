---
title: Batching and Invokers
one_liner: Each tracer.invoke() is one worker on its own [start, size] batch slice; empty invokes see the full batch; the Batcher narrows reads and widens writes; barriers synchronize cross-invoke value sharing.
tags: [concept, mental-model, batching, invokers]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/deferred-execution.md]
sources: [src/nnsight/intervention/tracer.py:336, src/nnsight/intervention/tracer.py:223, src/nnsight/intervention/batching.py:66, src/nnsight/intervention/batching.py:85, src/nnsight/intervention/barrier.py:35]
---

# Batching and Invokers

## What this is for

`tracer.invoke(...)` batches several inputs into one forward pass while running different intervention code on each. Each invoke becomes one `Mediator` (one greenlet worker) with a `batch_group = [start, size]`, so its interventions see only its own rows of every activation.

An **empty** invoke (`tracer.invoke()`, no args) has `batch_group = None` and sees the **whole** combined batch — useful for shared logic over all rows.

## When to use / when not to use

- Use multiple input invokes to give different inputs one shared forward (activation patching, ablations, batched comparison).
- Use an empty invoke to run logic over the full batch.
- A single input is enough? Just pass it to `trace("input")` — that's one implicit invoke.
- Batching **two or more** input invokes needs `_batch_size`/`_batch` on the model class. `TransformersModel` provides them; base `NNsight` supports a single invoke only.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace() as tracer:
    with tracer.invoke("Hello"):
        a = model.lm_head.output[:, -1].save()          # rows [0, 1)
    with tracer.invoke(["World", "Test foo bar"]):
        b = model.lm_head.output[:, -1].save()          # rows [1, 3)
    with tracer.invoke():
        allx = model.lm_head.output[:, -1].save()        # whole batch
```

Verified shapes:

```
a: (1, 50257)   b: (2, 50257)   all: (3, 50257)
```

## How an invoke becomes a worker

`Invoker` (`intervention/tracer.py:336`) is a `Tracer` subclass. Its `execute` doesn't run the model — it *registers* the invoke:

1. Its body is captured/compiled like any trace block (see [Deferred Execution](deferred-execution.md)).
2. `self.tracer.batcher.add(*args, **kwargs)` records the input and returns its `batch_group` (the batcher belongs to the outer tracer).
3. A `Mediator(code, glbls, lcls, node=..., shared=frame.f_locals)` is built with that `batch_group` and appended to `interleaver.mediators`.

The outer `InterleavingTracer.execute` (`intervention/tracer.py:223`) creates the `Batcher` (`self.batcher`) and runs the trace body once to collect all invokes, then hands the batcher to `Envoy.interleave`, which assembles it (→ the model's `_batch`) into the combined input, registers it on the interleaver for the run, and starts every worker.

`tracer.invoke(...)` while the model is already running raises (`Invoker.__init__`):

```
ValueError: Cannot invoke while the model is already running.
```

## Batcher: accumulate, then narrow/widen

`Batcher` (`batching.py:66`) lives on the interleaver for one trace.

### Accumulating

`add(*inputs, **kwargs)` (`batching.py:171`):

- `model._batch_size(...)` reports the input's row count. `0` (no data) → an empty invoke, `groups.append(None)`, returns `None`.
- Otherwise assign `group = [total, size]`, bump `total`, store the invoke, return the group.

### Per-fire narrow/widen

When a hook fires and `Interleaver.handle` serves a worker:

- **read**: `batcher.narrow(value, group)` (`batching.py:85`) slices every batched tensor (leading dim `== total`) down to `[start, start+size)`. Non-batched tensors and empty invokes pass through.
- **write**: `batcher.widen(full, group, edited)` (`batching.py:103`) splices the edited rows back into the full batch (via `cat`, keeping autograd correct).

`batcher.batching` is `True` only with **2+ input invokes** (`batching.py:183`). A lone invoke *is* the whole batch, so `narrow`/`widen` are no-ops — single-input traces pay no slicing overhead.

## Empty invoke semantics

`tracer.invoke()` with no arguments:

- Contributes no rows (`batch_group = None`), sees the full combined batch.
- Runs as its own worker, so it can access modules in an independent order relative to other invokes.
- Doesn't call `_batch`, so it works even on base `NNsight`.

At least one input invoke must exist, or `trace()` needs direct input — otherwise:

```
ValueError: trace() needs an input, or at least one `with tracer.invoke(...)` block
```

## Batched skip

If interventions `.skip()` a module inside a batched forward, there is no body output to splice into — the body didn't run. Each invoke's replacement is collected (`gather_skip`) and concatenated into the full-batch output (`assemble_skip`). Every invoke must skip the module (or none), since a shared forward can't run for only the unskipped rows:

```
ValueError: A batched `.skip()` has to cover every row: skip the module in every
invoke, or none — a shared forward can't run for only the rows an invoke left unskipped.
```

## Cross-invoke variable sharing

Blocks written in the same frame share their locals through the `Scope`'s `shared` dict (`tracing/util.py:32`), so a name bound in one invoke is visible in a later one. But workers resume in **model-reached** order, not definition order — so a value must be *bound before it's read*. When one block reads an activation and another writes it, use a **barrier**.

## Barriers: cross-invoke handoff on the same location

`tracer.barrier(n)` (`intervention/tracer.py:118`) returns a `Barrier` (`barrier.py:35`). Each block calls it; the first `n-1` park on `Event.BARRIER`; the last to arrive releases the rest by switching each parked worker directly. Everything above a barrier has happened before anything below one.

Verified — invoke 2 reuses invoke 1's embeddings:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in the city of"):
        embeddings = model.transformer.wte.output   # read
        barrier()
        sent = model.output.logits[:, -1].argmax(-1).save()
    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
        barrier()
        model.transformer.wte.output = embeddings    # write, after invoke 1's read
        received = model.output.logits[:, -1].argmax(-1).save()
# sent == received  ->  both [6342]
```

A barrier fewer blocks reach than it was built for never releases; the waiting blocks report it when the run ends.

## Order rules

- **Within an invoke:** access modules in forward-pass order (read `.input` before `.output`). Out-of-order raises `OutOfOrderError`.
- **Across invokes:** they share one forward; workers resume in the order the model reaches what each asked for.
- **Same module across invokes:** use a `barrier()` so the reader runs before the writer.

## Gotchas

- **`_batch_size`/`_batch` required for 2+ input invokes.** Base `NNsight` raises `NotImplementedError` on `_batch` with multiple invokes; use `TransformersModel`, implement them, or restructure as one input invoke + empty invokes.
- **A tensor is "batched" only if its leading dim equals the combined batch size.** `narrow`/`widen` leave others alone — a shape coincidence could in principle mislead them.
- **Custom batch layouts subclass `Batcher`** and override `narrow`/`widen`/`assemble`. vLLM's `VLLMBatcher` (`modeling/vllm/batching.py`) maps rows onto a flat token axis.
- **Empty invoke with no preceding input** has nothing to forward — provide an input invoke first.

## Related

- [Threading and Mediators](threading-and-mediators.md) — how workers execute, park, and synchronize.
- [Deferred Execution](deferred-execution.md) — how invoke bodies are captured and compiled.
- Source: `src/nnsight/intervention/tracer.py` (`Invoker`, `InterleavingTracer.execute`, `barrier`), `src/nnsight/intervention/batching.py` (`Batcher`), `src/nnsight/intervention/barrier.py` (`Barrier`).

---
title: Batching and Invokers
one_liner: Each tracer.invoke() is one worker on its own [start, size] batch slice; empty invokes see the full batch; the Batcher narrows reads and widens writes.
tags: [concept, mental-model, batching, invokers]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/deferred-execution.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/batching.py, src/nnsight/intervention/barrier.py]
---

# Batching and Invokers

## What this is for

`tracer.invoke(...)` batches several inputs into one forward pass while running different intervention code on each. Each invoke becomes one `Mediator` (one greenlet worker) with a `batch_group = [start, size]`, so its interventions see its own rows of every activation the batcher can scope — one whose leading dimension is the combined batch size, or a whole multiple of it. A value shaped any other way is served to every invoke whole, and a write to it warns.

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

`Invoker` (`intervention/tracer.py`) is a `Tracer` subclass. Its `execute` doesn't run the model — it *registers* the invoke:

1. Its body is captured/compiled like any trace block (see [Deferred Execution](deferred-execution.md)).
2. `self.tracer.batcher.add(*args, **kwargs)` records the input and returns its `batch_group` (the batcher belongs to the outer tracer).
3. A `Mediator(code, glbls, lcls, node=..., shared=frame.f_locals)` is built with that `batch_group` and appended to `interleaver.mediators`.

The outer `InterleavingTracer.execute` creates the `Batcher` (`self.batcher`) and runs the trace body once to collect all invokes, then hands the batcher to `Envoy.interleave`, which assembles it (→ the model's `_batch`) into the combined input, registers it on the interleaver for the run, and starts every worker.

`tracer.invoke(...)` while the model is already running raises (`Invoker.__init__`):

```
ValueError: Cannot invoke while the model is already running.
```

## Batcher: accumulate, then narrow/widen

`Batcher` lives on the interleaver for one trace.

### Accumulating

`Batcher.add(*inputs, **kwargs)`:

- `model._batch_size(...)` reports the input's row count. `0` (no data) → an empty invoke, `groups.append(None)`, returns `None`.
- Otherwise assign `group = [total, size]`, bump `total`, store the invoke, return the group.

### Per-fire narrow/widen

When a hook fires and `Interleaver.handle` serves a worker:

- **read**: `Batcher.narrow(value, group)` slices every scoped tensor down to the group's rows. A tensor is scoped when its leading dim is `total` (one row per row of batch, narrowed to `[start, start + size)`) or a whole multiple of it — `k = shape[0] // total` rows per row of batch, narrowed to `[start*k, size*k)`, which is the layout a model that folds tokens into the batch axis produces (`(batch*seq, hidden)` ahead of an MoE router). Anything else, and every empty invoke, passes through whole.
- **write**: `Batcher.widen(full, group, edited)` splices the edited rows back into the full batch (via `cat`, keeping autograd correct), under the same row rule. The replacement has to keep the group's row count — the splice takes it as given, so one of the wrong height builds a batch that is no longer the model's, and the mismatch surfaces in some later module or not at all. A replacement for a tensor no row rule scoped has nowhere to go and is dropped with a warning; an in-place edit to one never reaches `widen` at all, so the batcher remembers what it served whole and notices, at the next narrow or widen, that torch's version counter moved:

```
UserWarning: An in-place edit to `model.visual.merger.output` applies to every
invoke: its leading dimension (256) is neither the batch size (2) nor a multiple
of it, so nnsight served the whole batch rather than this invoke's rows.
```

A read of such a value does not warn: a value that is not batched at all — a causal mask, rotary `cos`/`sin` — is indistinguishable by shape from one batched on an axis the batcher cannot read, and most are the former.

`Batcher.batching` is `True` only with **2+ input invokes**. A lone invoke *is* the whole batch, so `narrow`/`widen` are no-ops — single-input traces pay no slicing overhead, and a lone invoke's write may change the leading dim and widen the run.

## Empty invoke semantics

`tracer.invoke()` with no arguments:

- Contributes no rows (`batch_group = None`), sees the full combined batch. The combined input is assembled from every invoke before any worker starts, so this holds wherever the empty invoke is written — first, last, or between two input invokes.
- Runs as its own worker, so it can access modules in an independent order relative to other invokes. Inside its own block, forward order still applies.
- Doesn't call `_batch`, so it works even on base `NNsight`.

At least one input invoke must exist, or `trace()` needs direct input. A trace with
neither is caught up front:

```
ValueError: trace() needs an input, or at least one `with tracer.invoke(...)` block
```

That guard counts blocks, not rows, so a trace holding *only* an empty invoke gets
past it and fails further in — see Gotchas.

## Batched skip

If interventions `.skip()` a module inside a batched forward, there is no body output to splice into — the body didn't run. Each invoke's replacement is collected (`gather_skip`) and concatenated into the full-batch output (`assemble_skip`). Every invoke must skip the module (or none), since a shared forward can't run for only the unskipped rows:

```
ValueError: A batched `.skip()` has to cover every row: skip the module in every
invoke, or none — a shared forward can't run for only the rows an invoke left unskipped.
```

## Cross-invoke variable sharing

Blocks written in the same frame share their locals through the `Scope`'s `shared` dict (`tracing/util.py`), so a name bound in one invoke is visible in another. Whether it is bound *yet* depends on where each worker has parked: a name is readable once the reader has parked at a location the model reaches after the binding. [usage/invoke-and-batching.md](../usage/invoke-and-batching.md#cross-invoke-value-sharing) states the rule and its two corollaries; a consumer that cannot park in between uses a **barrier**.

## Barriers: an ordered handoff

`InterleavingTracer.barrier(n)` returns a `Barrier`. Each block calls it; the first `n-1` park on `Event.BARRIER`; the last to arrive releases the rest by switching each parked worker directly. Everything above a barrier has happened before anything below one.

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

A barrier fewer blocks reach than it was built for never releases; `check_dangling_mediators` turns that into a `ValueError` when the run ends, so a mis-count is an error rather than a hang. A barrier *more* blocks reach releases early, and the block it let through raises `NameError` on the value it came for.

## Order rules

- **Within an invoke:** access modules in forward-pass order (read `.input` before `.output`). Out-of-order raises `OutOfOrderError`. An empty invoke is no exception; what it resets is its ordering relative to the *other* invokes.
- **Across invokes:** they share one forward; workers resume in the order the model reaches what each asked for.
- **Handing a value across:** a `barrier()` when the consumer cannot park past the producer, which includes every consumer whose first statement is a write.

## Gotchas

- **`_batch_size`/`_batch` required for 2+ input invokes.** Base `NNsight` raises `NotImplementedError` on `_batch` with multiple invokes; use `TransformersModel`, implement them, or restructure as one input invoke + empty invokes.
- **A tensor is scoped to an invoke only if its leading dim is the combined batch size or a whole multiple of it.** `narrow`/`widen` leave the rest alone. The multiple is what makes a flattened `(batch*seq, ...)` activation scopable, and it is a rule about shape, not provenance: a leading dim that is a multiple by coincidence — four experts against two invokes — is sliced as if it were rows, which is the accepted cost of covering the flattened case. A write to a tensor no rule scoped warns, naming the location, the leading dim and the batch size; a read does not.
- **Custom batch layouts subclass `Batcher`** and override `narrow`/`widen`/`assemble`. vLLM's `VLLMBatcher` (`modeling/vllm/batching.py`) maps rows onto a flat token axis.
- **A trace whose only invoke is empty has no rows to run.** On `TransformersModel` the tokenizer raises `IndexError: list index out of range`; on base `NNsight` the forward raises `TypeError` for its missing argument. Give the trace an input invoke or a direct input.

## Related

- [Threading and Mediators](threading-and-mediators.md) — how workers execute, park, and synchronize.
- [Deferred Execution](deferred-execution.md) — how invoke bodies are captured and compiled.
- Source: `src/nnsight/intervention/tracer.py` (`Invoker`, `InterleavingTracer.execute`, `barrier`), `src/nnsight/intervention/batching.py` (`Batcher`), `src/nnsight/intervention/barrier.py` (`Barrier`).

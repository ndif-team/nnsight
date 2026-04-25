---
title: Invoke and Batching
one_liner: Multiple inputs in one trace via `tracer.invoke(...)`, including empty invokes that operate on the full batch.
tags: [usage, batching, invoker]
related: [docs/usage/trace.md, docs/usage/barrier.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracing/invoker.py, src/nnsight/intervention/batching.py, src/nnsight/intervention/tracing/tracer.py:433, src/nnsight/intervention/interleaver.py:718]
---

# Invoke and Batching

## What this is for

`tracer.invoke(...)` adds an input to the trace's batch and runs an intervention block against that input's slice of the batch. Each invoke runs as a **separate worker thread** — they are serialized by the interleaver but each holds its own intervention code, history, and iteration counters.

There are two kinds of invokes:

- **Input invoke**: `tracer.invoke(prompt)` — contributes data to the batch. Calls `model._prepare_input` and (for the 2nd+ input invoke) `model._batch`. Gets `batch_group = [start, size]`.
- **Empty invoke**: `tracer.invoke()` — no input, operates on the **entire** combined batch. `batch_group = None`. Does not call `_batch`.

Source: `src/nnsight/intervention/batching.py:114` (Batcher), `src/nnsight/intervention/tracing/invoker.py:14` (Invoker), `src/nnsight/intervention/interleaver.py:718` (Mediator).

## When to use / when not to use

- Use multiple input invokes to run several prompts in one forward pass. Requires a model that implements `_prepare_input` and `_batch` (`LanguageModel` does — base `NNsight` does not).
- Use empty invokes to break a single input into multiple intervention threads (avoids forward-pass-order constraints within a single invoke), or to run different logic on the combined batch.
- Use a single positional arg on `.trace(...)` if you only have one input — it creates an implicit invoke.

## Canonical pattern

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        out_paris = model.lm_head.output[:, -1].save()

    with tracer.invoke("The Colosseum is in"):
        out_rome = model.lm_head.output[:, -1].save()
```

## Batched input (single invoke, list of strings)

```python
with model.trace(["Hello", "World"]):
    logits = model.lm_head.output.save()  # shape: [2, seq, vocab]
```

For `LanguageModel`, `_prepare_input` tokenizes the list with padding (`src/nnsight/modeling/language.py:241`).

## Empty invokes — operate on the full batch

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out_a = model.lm_head.output[:, -1].save()       # shape [1, vocab]
    with tracer.invoke(["World", "Test"]):
        out_b = model.lm_head.output[:, -1].save()       # shape [2, vocab]

    # Empty invoke: sees the merged batch (3 rows)
    with tracer.invoke():
        out_all = model.lm_head.output[:, -1].save()     # shape [3, vocab]

    # Another empty invoke is another thread — same merged batch
    with tracer.invoke():
        first_layer = model.transformer.h[0].output.save()
```

`Batcher.batch` returns `batch_group=None` for empty invokes (`batching.py:196`); during interleaving `narrow(None)` returns the full tensor and `swap(None, value)` replaces the full tensor.

## Out-of-order access via empty invokes

A single invoke's worker thread blocks on each `.output` request — modules must be accessed in forward-pass order. To access modules "out of order" use a second invoke (its thread starts at the top of the next forward pass... or, in single-pass mode, runs serially after the prior invoke):

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        layer5 = model.transformer.h[5].output.save()
    with tracer.invoke():       # empty invoke, separate thread
        layer1 = model.transformer.h[1].output.save()
```

## Cross-invoke variable sharing

Variables defined in one invoke are visible to later invokes (when `CONFIG.APP.CROSS_INVOKER` is `True`, the default). The mediator pushes its locals to the parent frame on every event (`interleaver.py:1304`, `Mediator.push/pull`).

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :]

    with tracer.invoke("The Colosseum is in"):
        # Use clean_hs from the previous invoke. If both invokes touch the
        # same module a barrier is required — see Gotchas.
        model.transformer.h[5].output[0][:, -1, :] = clean_hs
        patched = model.lm_head.output.save()
```

When two invokes both access the same module, you need `tracer.barrier(n)` to synchronize them. See `docs/usage/barrier.md`.

## Threading model

- One mediator (worker thread) per invoke.
- Threads run serially — `Interleaver.__enter__` starts each one, then drains its first event. Only one mediator's intervention code runs at a time.
- Each mediator owns: an `event_queue`/`response_queue` pair, a `history` set, an `iteration_tracker`, a `batch_group`, and a `hooks` list. See `Mediator` in `src/nnsight/intervention/interleaver.py:718`.
- Worker thread captures the caller's CUDA stream so it doesn't race with the main thread (`interleaver.py:900`).

## Implementing batching for a custom model

To support multiple input invokes on a non-`LanguageModel`, override `_prepare_input` and `_batch` (mixin: `Batchable` in `src/nnsight/intervention/batching.py:35`):

```python
from nnsight import NNsight

class MyModel(NNsight):
    def _prepare_input(self, *inputs, **kwargs):
        # return (args, kwargs, batch_size)
        return inputs, kwargs, len(inputs[0])

    def _batch(self, batched_input, *args, **kwargs):
        # combine new args/kwargs with batched_input -> (args, kwargs)
        ...
```

Without these, multiple input invokes raise `NotImplementedError: Batching is not implemented`.

## Gotchas

- Empty invokes do **not** trigger `_batch` — they always work, even on base `NNsight`.
- An empty invoke must be preceded by at least one input invoke (or a `.trace(input)`). Otherwise nothing has been put on the batch and the model never executes.
- Cannot nest invokes (`Cannot invoke during an active model execution / interleaving.`) — `Invoker.__init__` raises if interleaving is in progress (`invoker.py:32`).
- Sharing a variable derived from `.output` of a module that **both** invokes also touch requires `tracer.barrier(n)` — otherwise `NameError` because the second invoke runs before the first has produced the value. See `docs/usage/barrier.md`.
- Within an invoke, modules must be accessed in forward-pass order. To access "out of order" use additional invokes.

## Related

- `docs/usage/trace.md`
- `docs/usage/barrier.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/save.md`

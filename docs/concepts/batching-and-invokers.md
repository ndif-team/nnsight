---
title: Batching and Invokers
one_liner: Each tracer.invoke() spawns one worker thread on its own batch slice; empty invokes run on the combined batch; barriers synchronize cross-invoke value sharing on the same module.
tags: [concept, mental-model, batching, invokers]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/deferred-execution.md]
sources: [src/nnsight/intervention/tracing/invoker.py:14, src/nnsight/intervention/batching.py:35, src/nnsight/intervention/batching.py:114, src/nnsight/intervention/tracing/tracer.py:433, src/nnsight/intervention/tracing/tracer.py:551, src/nnsight/intervention/interleaver.py:1304]
---

# Batching and Invokers

## What this is for

`tracer.invoke(...)` is how you batch multiple inputs into a single forward pass while running different intervention code on each. Each invoke becomes one `Mediator` (one worker thread) and is assigned a `batch_group = [start, size]` so its worker only sees its own slice.

Empty invokes (`tracer.invoke()` with no arguments) are a special form that operates on the **entire** combined batch — useful for breaking up interventions or running shared logic at the end.

## When to use / when not to use

- Use multiple input invokes when you need different inputs to share one forward pass (typical for activation patching, ablations, batched comparison).
- Use empty invokes to access the same module twice in different code blocks (each invoke is a separate worker, so each can access modules in fresh forward-pass order — see [Threading and Mediators](threading-and-mediators.md)).
- Use a single invoke when one input is enough — the implicit invoke from `model.trace("input")` covers this case.
- Multiple input invokes require `_prepare_input` and `_batch` on your model class. `LanguageModel` provides these; base `NNsight` doesn't.

## Canonical pattern

```python
with model.trace() as tracer:
    # Two input invokes — each gets its own batch slice.
    with tracer.invoke("Hello"):
        a = model.lm_head.output[:, -1].save()  # batch_group = [0, 1]

    with tracer.invoke(["World", "Test"]):
        b = model.lm_head.output[:, -1].save()  # batch_group = [1, 2]

    # Empty invoke — operates on full batch [3].
    with tracer.invoke():
        all_logits = model.lm_head.output[:, -1].save()  # shape [3, vocab]
```

## How invokes become mediators

`Invoker` (`tracing/invoker.py:14`) is a `Tracer` subclass. Each `with tracer.invoke(...)`:

1. Captures the with-block source like a regular `Tracer`.
2. Compiles the body into an intervention function wrapped in try/catch (`Invoker.compile`):
   ```python
   def __nnsight_tracer_<id>__(__nnsight_mediator__, __nnsight_tracing_info__):
       __nnsight_mediator__.pull()
       try:
           # captured body
       except Exception as exception:
           __nnsight_mediator__.exception(exception)
       else:
           __nnsight_mediator__.end()
   ```
3. Calls `tracer.batcher.batch(model, *args, **kwargs)` (`batching.py:146`) to register the invoke's input and get its `batch_group`.
4. Constructs a `Mediator(fn, info, batch_group=batch_group)` and appends to `tracer.mediators`.

When the outer `with model.trace() as tracer:` block exits, `InterleavingTracer.execute` runs the compiled tracer function once on the main thread to populate `tracer.mediators`, then enters `Envoy.interleave(...)` which starts every mediator's worker.

## Batcher: input accumulation and slicing

`Batcher` (`batching.py:114`) lives on the tracer for the duration of one trace. Two phases:

### Phase 1: accumulating inputs

For each `tracer.invoke(*args, **kwargs)`:

- If args or kwargs were provided, `model._prepare_input(*args, **kwargs)` normalizes them and returns `(args, kwargs, batch_size)`.
- The first input invoke stores its prepared input directly. Subsequent input invokes call `model._batch((batched_args, batched_kwargs), *new_args, **new_kwargs)` to merge.
- Each input invoke gets a `batch_group = [start_idx, batch_size]`.
- Empty invokes (no args, no kwargs) get `batch_group = None`.

### Phase 2: per-fire narrowing

When the worker requests a value and the hook fires, `batcher.narrow(mediator.batch_group)` slices the activation tensor along dim 0. When the worker assigns, `batcher.swap(batch_group, value)` splices back. With `batch_group = None`, both pass through the full batch.

`batcher.needs_batching` becomes True only when there are **2+ input invokes** — single-input traces skip the narrow/swap overhead.

## Empty invoke semantics

`tracer.invoke()` with no arguments:

- Does **not** call `_batch()` — works on base `NNsight` even without batching support.
- Has `batch_group = None` — sees the full combined batch.
- Runs as a separate worker thread, so it can access modules in independent forward-pass order.

A trace can mix one input invoke with any number of empty invokes regardless of batching support:

```python
# Works on base NNsight (no _batch implementation needed).
with model.trace() as tracer:
    with tracer.invoke(input):
        a = model.layer.output.save()
    with tracer.invoke():  # empty, runs on full batch
        b = model.other_layer.output.save()
```

At least one input invoke must precede any empty invoke — without input, the model has nothing to run.

## Cross-invoke variable sharing

By default (`CONFIG.APP.CROSS_INVOKER = True`), variables from one invoke can be referenced in a later invoke:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        embeddings = model.transformer.wte.output  # captured

    with tracer.invoke("World"):
        model.transformer.wte.output = embeddings  # used
```

How it works (`interleaver.py:1304`):

- After every event, `Mediator.push()` walks the worker's frame locals and writes them into the user's calling frame.
- Before every event, `Mediator.pull()` re-reads from the calling frame.
- This works because mediators run serially — by the time invoke 2's worker starts, invoke 1's worker has already pushed its locals.

Disable with `CONFIG.APP.CROSS_INVOKER = False` to make each invoke isolated. Useful for debugging.

## Barriers: when both invokes touch the same module

If two invokes both access the same module's `.output`, they each need that module's hook to fire — but on a single forward pass, the hook fires once. The first invoke's worker sees it; the second invoke's worker has already requested it and will be denied (`OutOfOrderError`) unless synchronized.

`tracer.barrier(n)` (`tracing/tracer.py:551`) returns a callable that pauses participating mediators until `n` of them have hit it:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)

    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[:, -1, :]
        barrier()  # invoke 1 reaches here, parks

    with tracer.invoke("The Colosseum is in"):
        barrier()  # invoke 2 hits, both released
        model.transformer.h[5].output[:, -1, :] = clean_hs  # now defined
        out = model.lm_head.output.save()
```

Implementation (`Mediator.handle_barrier_event`, `interleaver.py:1123`): the second mediator to reach the barrier is the one with all `n` participant names. It releases the parked mediators by calling their `.respond()`. Variables pushed by invoke 1 before its barrier are now visible to invoke 2.

## Order rules

- **Within an invoke**: access modules in forward-pass order. Out-of-order access deadlocks (`OutOfOrderError`).
- **Across invokes**: invokes run in the order you define them. The model's forward pass happens once for all of them.
- **Same module across invokes**: requires a `barrier()` to share values, otherwise the variable from invoke 1 is not yet pushed when invoke 2 tries to use it.

## Gotchas

- **`tracer.invoke(*args)` inside an active trace raises.** The check at `Invoker.__init__` (`tracing/invoker.py:32`) prevents nested invokes.
- **`NotImplementedError: Batching is not implemented`** when calling `_batch` on base `Envoy`. Either use `LanguageModel`, implement `_prepare_input` / `_batch`, or restructure as one input invoke + empty invokes.
- **Diffusion models use `DiffusionBatcher`** (`batching.py:325`) which scales batch groups by `num_images_per_prompt` and handles guidance (2x batch). Override `_batcher_class` on your model to plug in a custom batcher.
- **Empty invoke without a preceding input invoke fails.** The model has nothing to forward.
- **Persistent cache hooks key on `mediator.batch_group`.** If `batch_group[0] == -1` (vLLM unscheduled request), the hook is a no-op for that mediator — see `cache_output_hook` in `hooks.py`.

## Related

- [Threading and Mediators](threading-and-mediators.md) — how invoke threads execute and sync.
- [Deferred Execution](deferred-execution.md) — how invoke bodies are captured and compiled.
- Source: `src/nnsight/intervention/tracing/invoker.py` (`Invoker`), `src/nnsight/intervention/batching.py` (`Batcher`, `DiffusionBatcher`, `Batchable`), `src/nnsight/intervention/tracing/tracer.py` (`InterleavingTracer.invoke`, `barrier`).

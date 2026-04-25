---
title: Execution Complete But Value Was Not Provided
one_liner: "Raised by check_dangling_mediators when the model finished but a mediator is still waiting for a module's value."
tags: [error, execution-order, dangling-mediator]
related: [docs/errors/missed-provider-error.md, docs/errors/out-of-order-error.md, docs/usage/iter.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/interleaver.py:652, src/nnsight/intervention/interleaver.py:667, src/nnsight/intervention/interleaver.py:677]
---

# Execution Complete But Value Was Not Provided

## Symptom

Raised as a `Mediator.MissedProviderError` (which is an `Exception`) at the end of the forward pass:

```
Execution complete but `model.transformer.h.5.output.i0` was not provided. Did you call an Envoy out of order? Investigate why this module was not called.
```

Variant emitted as a `UserWarning` instead of an exception when the unsatisfied request happened beyond the first iteration of a generator loop (e.g., `.iter[:]` after generation finished):

```
UserWarning: Execution complete but `<requester>` was not provided. If this was in an Iterator at iteration <N> this iteration did not happen. If you were using `.iter[:]`, this is likely not an error.
```

The `<requester>` is a dotted path of the form `<envoy.path>.<key>.i<iteration>` — for example `model.transformer.h.5.output.i0` means "layer 5's output on iteration 0".

## Cause

After the model's forward pass returns, `Envoy.interleave` calls `Interleaver.check_dangling_mediators` (`src/nnsight/intervention/interleaver.py:608` calls into `:652`). For every mediator that is still `alive` (still has a worker thread blocked on a request), the interleaver:

1. Pulls the pending event off the mediator's `event_queue` to recover the requester string.
2. Calls `mediator.respond(Mediator.MissedProviderError(...))` to inject the exception into the waiting worker (`:667`).
3. If that mediator is at iteration > 0 (inside a generator step that never came), it downgrades the second-round signal to a `warnings.warn` (`:677`) because hitting this in `.iter[:]` is the documented end-of-generation case, not user error.

The exception is then re-raised in the user's call stack via the worker-thread join in `Mediator.handle_exception_event` (`src/nnsight/intervention/interleaver.py:1119`).

## Common triggers

- Requesting `.output` on a module path that exists in `print(model)` but is **not called** on this input (model dispatches between two branches; only one branch fires).
- Code placed after an unbounded `for step in tracer.iter[:]:` loop — the iterator never yields a final batch, so anything past the loop is left waiting.
- Using `tracer.iter[N]` for an `N` larger than the number of steps actually generated (early EOS, `max_new_tokens` smaller than `N+1`).
- Skipping a module via `module.skip(value)` and then trying to read `.output` on a child of the skipped module — child modules don't fire.
- Constructing a swap (`module.output = ...`) on a module that the model never calls.

## Fix

When the trigger is unbounded iter:

```python
# WRONG — `final` never gets fulfilled; check_dangling_mediators raises
with model.generate("Hi", max_new_tokens=3) as tracer:
    for step in tracer.iter[:]:
        hs = model.transformer.h[-1].output.save()
    final = model.lm_head.output.save()
```

```python
# FIXED — split the post-iter request into an empty invoke or use a bounded slice
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke("Hi"):
        for step in tracer.iter[:]:
            hs = model.transformer.h[-1].output.save()
    with tracer.invoke():
        final = model.lm_head.output.save()
```

When the trigger is a path that does not execute, validate first:

```python
with model.scan("Hi"):
    # raises here if the path is wrong / not called
    nnsight.save(model.transformer.h[5].output[0].shape)
```

## Mitigation / how to avoid

- Always reach unbounded iteration via a separate empty invoke for any "after-generation" code.
- Prefer `tracer.iter[:N]` (bounded) over `tracer.iter[:]` when you know the step count.
- Use `model.scan(input)` to confirm a module fires before relying on it inside a real trace.
- The warning variant (iter-after-generation) is benign and silenceable with `warnings.filterwarnings("ignore")` on `UserWarning` if it's noisy in your pipeline.

## Related

- `docs/errors/missed-provider-error.md` (base class)
- `docs/errors/out-of-order-error.md` (subclass for already-seen providers)
- `docs/usage/iter.md`
- `docs/concepts/threading-and-mediators.md`

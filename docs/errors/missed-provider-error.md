---
title: MissedProviderError
one_liner: "Mediator.MissedProviderError: Execution complete but `<requester>` was not provided — a value request was never satisfied by any provider. The primary post-refactor error class for execution-order failures."
tags: [error, execution-order, threading]
related: [docs/errors/out-of-order-error.md, docs/errors/value-was-not-provided.md, docs/concepts/threading-and-mediators.md, docs/usage/iter-all-next.md]
sources: [src/nnsight/intervention/interleaver.py:753, src/nnsight/intervention/interleaver.py:667, src/nnsight/intervention/interleaver.py:652]
---

# MissedProviderError

> **Primary error class.** Post-`refactor/transform`, the main failure mode users see when a value isn't delivered is `MissedProviderError`. The classic `OutOfOrderError` is now a **subclass** of `MissedProviderError` (see [out-of-order-error.md](out-of-order-error.md)) — it covers the early-detection case where nnsight already saw the provider fire and consumed it, so it can answer "out of order" immediately rather than waiting until forward finishes. Same root cause, different code paths.

## Symptom

Two surface forms exist depending on where the request was rejected.

When the model finishes running but a mediator is still waiting for a value (raised by `Interleaver.check_dangling_mediators`):

```
ValueError: Execution complete but `model.transformer.h.5.output.i0` was not provided. Did you call an Envoy out of order? Investigate why this module was not called.
```

Inside a generation loop, the warning variant:

```
UserWarning: Execution complete but `<requester>` was not provided. If this was in an Iterator at iteration <N> this iteration did not happen. If you were using `.iter[:]`, this is likely not an error.
```

When a request collides with an already-seen provider (raised by the value-event handler):

```
Mediator.OutOfOrderError: Value was missed for <requester>. Did you call an Envoy out of order?
```

## Cause

`MissedProviderError` is the base class for any "your worker thread asked for a value that nothing produced" condition (`src/nnsight/intervention/interleaver.py:753`). Two paths raise it:

1. **Dangling-mediator path** (`Interleaver.check_dangling_mediators`, `src/nnsight/intervention/interleaver.py:652`): after the model finishes its forward pass, the interleaver checks every mediator. If any is still alive (still waiting for a value), it sends a `MissedProviderError` into the worker thread so the user-side raises with a clear message.
2. **Out-of-order path** (`Mediator.handle_value_event`, `src/nnsight/intervention/interleaver.py:1049`): if a request for `requester` is made after `requester` has already been satisfied, the mediator returns `OutOfOrderError` (a subclass of `MissedProviderError`).

The "iteration N did not happen" variant is downgraded to a `warnings.warn` because unbounded iterators (`tracer.iter[:]`, `tracer.all()`) deliberately end up dangling once generation stops — that is expected behavior, not user error.

## Common triggers

- Accessing `.output` / `.input` on a module that the model never calls during this forward pass (e.g., a model dispatches between two implementations and you picked the path not taken).
- Typo or stale path on a module that doesn't actually fire (e.g., requesting `model.transformer.h[100]` on a 12-layer model — earlier `IndexError` usually catches this, but a path that resolves but is not called slides through).
- Using `tracer.iter[N]` for a step that the model stops generating before reaching (e.g., due to EOS).
- A skipped module: `module.skip(value)` was called and you also tried to read its `.output` directly.
- Code after `for step in tracer.iter[:]:` — the unbounded iterator never returns control, so subsequent requests are stranded. See [unbounded-iter](../gotchas/) and `docs/usage/iter-all-next.md`.

## Fix

Sanity-check the module path actually executes:

```python
with model.scan("Hello"):
    print(model.transformer.h[5].output.shape)  # raises if path doesn't fire
```

For unbounded iter:

```python
# WRONG — code after iter[:] is unreachable; final_logits never resolves -> MissedProviderError
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[:]:
        hidden = model.transformer.h[-1].output.save()
    final_logits = model.lm_head.output.save()
```

```python
# FIXED — separate empty invoke runs after generation; or use bounded iter[:3]
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke("Hello"):
        for step in tracer.iter[:]:
            hidden = model.transformer.h[-1].output.save()
    with tracer.invoke():
        final_logits = model.lm_head.output.save()
```

## Mitigation / how to avoid

- Use `model.scan(input)` to validate that every module path you reference actually executes.
- Prefer bounded `tracer.iter[:N]` over `tracer.iter[:]` whenever the step count is known.
- Catch the base class to handle both subclasses: `except Mediator.MissedProviderError`.

## Related

- `docs/errors/out-of-order-error.md`
- `docs/errors/value-was-not-provided.md`
- `docs/concepts/threading-and-mediators.md`
- `docs/usage/iter-all-next.md`

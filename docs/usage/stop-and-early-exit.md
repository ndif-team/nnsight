---
title: Stop and Early Exit
one_liner: Cut a forward pass short with `tracer.stop()`; raises `EarlyStopException`, swallowed by the interleaver.
tags: [usage, control-flow, early-stop]
related: [docs/usage/trace.md, docs/usage/iter.md, docs/usage/skip.md]
sources: [src/nnsight/intervention/tracing/tracer.py:447, src/nnsight/intervention/interleaver.py:355, src/nnsight/intervention/interleaver.py:1264, src/nnsight/intervention/interleaver.py:589]
---

# Stop and Early Exit

## What this is for

`tracer.stop()` aborts the current forward pass at the point where the worker thread is currently blocked. Any module that has not yet executed by the time `stop()` fires is **never executed** — the model raises `EarlyStopException`, the interleaver catches it on `__exit__`, and the trace exits cleanly.

This is how you "save what you need and bail" without dragging the rest of the forward pass along.

## When to use / when not to use

- Use to short-circuit when you have already collected the activations you need.
- Use to terminate generation mid-step when a condition is met.
- Don't use as an error path — `stop()` is treated as a successful early exit, not as an error.
- Don't use to skip a single module — that's `module.skip(value)` (`docs/usage/skip.md`).

## Canonical pattern

```python
with model.trace("Hello") as tracer:
    h0 = model.transformer.h[0].output[0].save()
    # We don't need anything past layer 0
    tracer.stop()

# Layers 1..N never ran. h0 is populated.
print(h0.shape)
```

## How it works

`InterleavingTracer.stop` (`src/nnsight/intervention/tracing/tracer.py:447`) calls `Mediator.stop()` (`interleaver.py:1264`):

```python
def stop(self):
    self.push()
    raise EarlyStopException()
```

`EarlyStopException` (`interleaver.py:355`) propagates up through the worker thread, then through the model's forward pass, and is finally **swallowed** by `Interleaver.__exit__` (`interleaver.py:589`):

```python
if exc_type is not None and issubclass(exc_type, EarlyStopException):
    return True
```

Mediator state is cleaned up via `cancel()` and registered hooks are removed via `Mediator.remove_hooks()`.

## Stop in generation

```python
with model.generate("Hello", max_new_tokens=20) as tracer:
    for step in tracer.iter[:]:
        tok = model.lm_head.output[0, -1].argmax(dim=-1).save()
        if tok.item() == model.tokenizer.eos_token_id:
            tracer.stop()
```

`stop()` ends the entire generation, not just one step. To skip a single step's interventions but keep generating, just exit the `if` block.

## Stop in nested invokes

`tracer.stop()` aborts the **current** mediator's interleaving. The mediator is the one whose worker thread is calling `stop()` (i.e. the current invoke). Other mediators that have already finished their work still get their saved values back.

## Saving state before stop

`Mediator.stop` calls `self.push()` first (`interleaver.py:1267`), which flushes the worker frame's locals back into the user's frame. So variables you assigned before `stop()` survive — even ones not explicitly `.save()`'d, as long as they go through the cross-invoker push pathway.

For values you intend to consume after the trace exits, **always** call `.save()` (or `nnsight.save(...)`) before `stop()`:

```python
with model.trace("Hello") as tracer:
    h = model.transformer.h[0].output[0].save()   # <-- save first
    tracer.stop()
```

## Gotchas

- `stop()` only works inside an active interleaving — calling it outside a trace raises because `interleaver.current` is `None`.
- Code after `stop()` in the same invoke does not run (Python sees the raised exception). Don't rely on side effects after the call.
- Anything that depends on a later module's output (e.g. `model.lm_head.output.save()` if you stop at layer 0) will not be populated and will hit `MissedProviderError` if requested.
- `EarlyStopException` is **not** an error — `Interleaver.__exit__` swallows it. Do not put `try/except EarlyStopException` around your trace expecting to catch user errors.
- For per-module bypass without aborting the whole forward pass, use `module.skip(value)` instead.

## Related

- `docs/usage/trace.md`
- `docs/usage/iter.md`
- `docs/usage/skip.md`
- `docs/usage/save.md`

---
title: Skip
one_liner: Bypass a module's forward computation with `module.skip(replacement)`.
tags: [usage, intervention, skip]
related: [docs/usage/access-and-modify.md, docs/usage/stop-and-early-exit.md, docs/usage/edit.md]
sources: [src/nnsight/intervention/envoy.py:450, src/nnsight/intervention/interleaver.py:1154, src/nnsight/intervention/interleaver.py:528]
---

# Skip

## What this is for

`module.skip(replacement)` tells the interleaver: when the model is about to call this module, **don't** execute its forward — return `replacement` as the output instead. Useful for ablating an entire submodule, swapping in a replacement (e.g. SAE-reconstructed activation), or routing around a buggy custom layer.

Implemented by injecting a magic `__nnsight_skip__` kwarg into the module's call. The wrapper installed by `Interleaver.wrap_module` checks for it and shortcuts the original forward (`src/nnsight/intervention/interleaver.py:528`).

## When to use / when not to use

- Use to ablate or replace a single module's contribution.
- Use to inject an externally-computed value at a specific point.
- Use `tracer.stop()` to abort the entire forward pass — `skip` only bypasses one module.
- Use `model.edit(...)` to make a skip persistent across all future traces.

## Canonical pattern

```python
with model.trace("Hello"):
    layer0_out = model.transformer.h[0].output

    # Use layer 0's output as layer 1's output (i.e., skip layer 1 entirely)
    model.transformer.h[1].skip(layer0_out)

    layer1_out = model.transformer.h[1].output.save()

# layer1_out equals layer0_out — layer 1 never ran
```

## Skip with a constructed replacement

```python
import torch

with model.trace("Hello"):
    inp = model.transformer.h[3].input
    # Replace layer 3's output with zeros of the same shape as the input
    model.transformer.h[3].skip((torch.zeros_like(inp), None))
```

The shape of `replacement` must match what the module would normally return. For modules that return a tuple, pass a tuple.

## How it works

`Envoy.skip` (`src/nnsight/intervention/envoy.py:450`) calls `interleaver.current.skip(...)` which sends a SKIP event to the mediator (`Mediator.skip`, `interleaver.py:1271`). When the mediator's `handle_skip_event` matches the requester (`interleaver.py:1154`):

```python
_, kwargs = self.interleaver.batcher.current_value
kwargs["__nnsight_skip__"] = value
self.respond()
```

The wrapper `nnsight_forward` installed on every wrapped module (`interleaver.py:528`) checks for the kwarg:

```python
def nnsight_forward(*args, **kwargs):
    if "__nnsight_skip__" in kwargs:
        return kwargs.pop("__nnsight_skip__")
    ...
```

So the original forward is never invoked, and the replacement is returned as the module output.

## Constraints

- `skip` requires `.input` (it triggers the input pre-hook to capture the live `(args, kwargs)` for kwarg injection). Calling `skip` on a module that has not been called yet, with no upstream module having executed, may not work — `requires_input` fires the same hook chain that `module.input` does (`envoy.py:450`).
- You cannot access **inner submodules** of a skipped module — they never execute, so their hooks never fire and any request for their `.output` will deadlock or raise `MissedProviderError`.
- Skips must respect forward-pass execution order, like all other accesses within a single invoke.
- The skip is one-shot per access — if the module is called again (e.g. across generation steps), each call needs its own skip if you want every step bypassed. Use `tracer.iter[...]` or persistent edits for that pattern.

## Persistent skip via `model.edit`

```python
with model.edit(inplace=True):
    inp = model.transformer.h[1].input
    model.transformer.h[1].skip(inp)  # always skip layer 1, pass-through

# Now every trace skips layer 1
with model.trace("Hello"):
    out = model.lm_head.output.save()
```

See `docs/usage/edit.md`.

## Gotchas

- Replacement shape / type must match what the module would return. A mismatched type causes downstream errors in the model's forward.
- `skip` is sent as a SKIP event; if you try to access `.output` of inner submodules of the skipped module, you'll get an `OutOfOrderError` or `MissedProviderError` because they never executed.
- `skip` can only be called inside an active trace — `Envoy.skip` is decorated with `@trace_only` (`envoy.py:450`).
- The kwarg shortcut means custom forward functions that *already* accept a `__nnsight_skip__` kwarg will collide. Don't name your kwargs that.

## Related

- `docs/usage/access-and-modify.md`
- `docs/usage/stop-and-early-exit.md`
- `docs/usage/edit.md`

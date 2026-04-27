---
title: OutOfOrderError
one_liner: "Mediator.OutOfOrderError: Value was missed for <requester> — module accessed out of forward-pass order within a single invoke."
tags: [error, execution-order, threading]
related: [docs/errors/missed-provider-error.md, docs/errors/value-was-not-provided.md, docs/concepts/threading-and-mediators.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/interleaver.py:760, src/nnsight/intervention/interleaver.py:1049, src/nnsight/intervention/interleaver.py:1181]
---

# OutOfOrderError

## Symptom

```
Mediator.OutOfOrderError: Value was missed for model.transformer.h.5.output.i0. Did you call an Envoy out of order?
```

The same surface text can appear from a `Setting ... is out of scope` swap path:

```
ValueError: Setting model.transformer.h.5.output.i0 is out of scope for scope <provider>. Did you call an Envoy out of order?
```

## Relationship to MissedProviderError

In `refactor/transform`, `OutOfOrderError` is a **subclass of `Mediator.MissedProviderError`** (see `src/nnsight/intervention/interleaver.py:760`). They surface in **different code paths** but have the **same root cause**: the value you asked for is not (or no longer) available.

```python
class MissedProviderError(Exception): ...
class OutOfOrderError(MissedProviderError): ...
```

- `OutOfOrderError` is the **eager** detection: the mediator already saw a provider with that requester string fire, so it raises immediately when the request arrives (`src/nnsight/intervention/interleaver.py:1049`). This catches the "ask for layer 1 after layer 5" pattern at the moment of asking.
- `MissedProviderError` (its parent) is the **late** detection: the model finished, the worker is still waiting, and `check_dangling_mediators` raises (`src/nnsight/intervention/interleaver.py:652`). This catches the "module didn't fire" or "iter step never happened" patterns where the mediator never saw the provider at all.

**Catching both at once:** `except Mediator.MissedProviderError` covers both. Legacy code or tutorials referring to "OutOfOrderError" still resolve to the same exception class.

`MissedProviderError` is the **primary** error class post-refactor; `OutOfOrderError` is its eager-detection subclass.

## Cause

Each invoke runs its body in a **single worker thread** that synchronizes with the model's forward pass. When the worker requests `module.output`, it blocks until the model fires that hook. The mediator tracks every provider it has already seen in `Mediator.history`; if a request arrives for a provider that was already seen and consumed, the mediator answers the worker with `OutOfOrderError` (`src/nnsight/intervention/interleaver.py:1049`).

This means **modules within a single invoke must be accessed in forward-pass order**. Asking for layer 1's output after layer 5 has already run is impossible — layer 1's value has already been delivered and discarded.

The `.i0` / `.i1` suffix on the requester string is the iteration counter (which generation step the request targets). On a single trace this is always `.i0`.

## Common triggers

- Accessing modules in reverse order inside a single invoke or trace body.
- Trying to read the same module's `.output` twice in one invoke.
- Reading a `.grad` for an early layer before later layers in a `with tensor.backward():` block (gradients flow in reverse, so access order also reverses — see `docs/usage/backward-and-grad.md`).
- Calling `module.skip(value)` in the wrong order so the skip handler sees the requester after its provider has passed (`src/nnsight/intervention/interleaver.py:1181`).

## Fix

```python
# WRONG — layer 5 fires before layer 1 inside the same invoke; deadlock / OutOfOrderError
with model.trace("Hello"):
    out5 = model.transformer.h[5].output.save()
    out1 = model.transformer.h[1].output.save()
```

```python
# FIXED — access modules in forward-pass order
with model.trace("Hello"):
    out1 = model.transformer.h[1].output.save()
    out5 = model.transformer.h[5].output.save()
```

To genuinely access modules out of forward order, run a second forward pass via an empty invoke (each invoke is its own worker thread, so they're independent):

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out5 = model.transformer.h[5].output.save()
    with tracer.invoke():           # empty invoke = new pass on the same batch
        out1 = model.transformer.h[1].output.save()
```

## Mitigation / how to avoid

- Lay your intervention code out top-to-bottom in the same order modules execute in `print(model)`.
- For backward passes, mirror the forward order in reverse inside the `with tensor.backward():` block.
- If you need both early and late activations and the access patterns interleave, split into multiple invokes.

## Related

- `docs/errors/missed-provider-error.md`
- `docs/errors/value-was-not-provided.md`
- `docs/concepts/threading-and-mediators.md`
- `docs/usage/invoke-and-batching.md`

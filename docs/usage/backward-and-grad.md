---
title: Backward and Gradients
one_liner: with tensor.backward(): runs a separate interleaving session in which .grad on tensors is hookable.
tags: [usage, backward, gradients]
related: [docs/usage/access-and-modify.md, docs/usage/save.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/tracing/backwards.py:81, src/nnsight/intervention/tracing/backwards.py:10, src/nnsight/__init__.py:158]
---

# Backward and Gradients

## What this is for

`with tensor.backward():` runs a **second, independent interleaving session** layered on top of `torch.Tensor.backward`. nnsight monkey-patches `Tensor.backward` at import time (`src/nnsight/__init__.py:177`); when called as a `with`-statement context, it builds a `BackwardsTracer` that captures the body, registers gradient hooks, and runs `tensor.backward(...)` under a `Patch` that exposes `.grad` as a hookable property on every tensor (`src/nnsight/intervention/tracing/backwards.py:10`).

Inside the backward block, the **only** thing you can request is `.grad` on a tensor you defined earlier. Module-level `.input` / `.output` are not available — that session is over by the time autograd runs.

## When to use / when not to use

- Use to read or modify the gradient of a specific tensor during backprop.
- Use for gradient-based attribution or guided optimization through a frozen model.
- Skip if you only need the loss value — plain `loss = ...; loss.backward()` (no `with`) still works because the patch falls back to vanilla backward when no `with` block is captured (`src/nnsight/__init__.py:168`).

## Canonical pattern

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)

    logits = model.lm_head.output
    loss   = logits.sum()

    # New, separate interleaving session — only .grad on tensors works inside.
    with loss.backward():
        grad = hs.grad.save()

print(grad.shape)
```

## Variations

### Modify a gradient

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    logits = model.lm_head.output

    with logits.sum().backward():
        hs_grad = hs.grad.save()        # read
        hs.grad[:] = 0                  # in-place modify
```

### Multiple backward passes — `retain_graph=True`

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    logits = model.lm_head.output

    with logits.sum().backward(retain_graph=True):
        grad1 = hs.grad.save()

    modified = logits * 2
    with modified.sum().backward():
        grad2 = hs.grad.save()
```

### Standalone backward (outside a `model.trace()`)

`with tensor.backward():` works on its own. Save the tensors you want gradients for during a previous trace, then open a backward block:

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    hs = hs.save()
    logits = model.lm_head.output.save()

# Outside any trace — backward gets its own interleaving session.
loss = logits.sum()
with loss.backward():
    grad = hs.grad.save()
```

## How `.grad` access works

Internally, `wrap_grad` returns a `property(getter, setter)` (`src/nnsight/intervention/tracing/backwards.py:10`). Reading `tensor.grad` registers a one-shot `register_hook` on the tensor (keyed by `id(tensor)`); when autograd fires the hook with the actual gradient, `interleaver.handle(f"{id(tensor)}.grad", grad)` delivers it to the worker thread. Writing `tensor.grad = value` issues a swap into that same channel.

Because the request key is `f"{id(tensor)}.grad"`, gradient errors look like `<id>.grad` rather than a module path — the requester is the tensor itself.

## Gotchas

- **Get any `.output` / `.input` BEFORE entering `with tensor.backward():`.** Inside the backward block, requesting anything other than `.grad` raises `ValueError("Cannot request ... in a backwards tracer. You can only request .grad.")` (`src/nnsight/intervention/tracing/backwards.py:73`).
- **Access gradients in REVERSE module order.** Autograd flows backward through the model. If you want gradients on layers 5 and 10 (forward order), request `layer10.grad` first, then `layer5.grad` — the same lockstep rule that applies to `.output` in forward order applies to `.grad` in reverse (`src/nnsight/intervention/tracing/backwards.py:69`).
- **`tensor.requires_grad_(True)` must be called before backward** if you want a gradient on a non-leaf intermediate tensor. nnsight does not auto-enable gradients.
- **`with tensor.backward():` only works on a tensor that was defined inside a captured tracing context** (so the backward AST can be extracted). Calling `with some_random_tensor.backward():` outside any `with`-block falls through to the vanilla `Tensor.backward` and returns its result instead of a tracer (`src/nnsight/__init__.py:163`).
- **Hooks on tensors are one-shot.** Each `.grad` access re-registers via `tensor.register_hook`. Re-entering the backward context re-registers them automatically.
- See [docs/gotchas/backward.md](../gotchas/backward.md) for the full set.

## Related

- [access-and-modify](access-and-modify.md) — Module-level `.output` / `.input` (forward-only).
- [trace](trace.md) — Forward-pass tracing.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — Mediator / interleaver model (BackwardsMediator is a subclass).

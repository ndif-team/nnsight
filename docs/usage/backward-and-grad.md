---
title: Backward and Gradients
one_liner: with tensor.backward(): runs the real backward interleaved so .grad on tensors is readable and editable.
tags: [usage, backward, gradients]
related: [docs/usage/access-and-modify.md, docs/usage/save.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/backward.py, src/nnsight/__init__.py]
---

# Backward and Gradients

## What this is for

`with tensor.backward():` runs the real backward pass **interleaved** with the body
of the `with` block, so the block can read and replace the `.grad` of any tensor as
the gradient reaches it. nnsight patches `torch.Tensor.backward` at import time; a
bare `tensor.backward()` (no `with` block) falls through to vanilla PyTorch
unchanged.

A backward trace is almost always **nested inside a forward trace**, so the tensors
whose gradients you want are the real ones produced during the run.

## When to use / when not to use

- Use to read or modify the gradient of a specific tensor during backprop.
- Use for gradient-based attribution or optimization through a frozen model.
- Skip if you only need the loss — plain `loss.backward()` (no `with`) still works.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello world"):
    hs   = model.transformer.h[-1].output      # a tensor
    loss = model.output.logits.sum()

    with loss.backward():                      # real backward, interleaved
        g = hs.grad.clone().save()             # read the gradient flowing into hs
        hs.grad = hs.grad * 2                  # ...and replace it downstream

print(g.shape)   # torch.Size([1, 2, 768])
```

Reading `hs.grad` parks the block until autograd produces that gradient; assigning
`hs.grad = ...` replaces the gradient that flows onward.

## Variations

### Read gradients in reverse-forward order

Gradients flow backward through the model, so request `.grad` in the reverse of the
forward order — later layers first:

```python
with model.trace("Hello world"):
    early = model.transformer.h[0].output
    late  = model.transformer.h[-1].output
    loss  = model.output.logits.sum()
    with loss.backward():
        g_late  = late.grad.clone().save()     # last layer's grad flows first
        g_early = early.grad.clone().save()     # then the first layer's
```

### Multiple backward passes — `retain_graph=True`

```python
with model.trace("Hello world"):
    hs     = model.transformer.h[-1].output
    logits = model.output.logits
    with logits.sum().backward(retain_graph=True):
        g1 = hs.grad.clone().save()
    with (logits.sum() * 2).backward():
        g2 = hs.grad.clone().save()
# g2 == 2 * g1
```

### Standalone backward (plain tensors)

`with tensor.backward():` works on its own for tensors whose autograd graph is
still alive:

```python
import torch
x = torch.tensor([2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
with loss.backward():
    g = x.grad.save()
# g == tensor([4., 6.])
```

## How `.grad` access works

For the duration of one backward run, `torch.Tensor.grad` is replaced by a
property. Reading `t.grad` registers a self-removing autograd hook on `t` (once per
tensor) and parks the block on the location `f"{id(t)}.grad"`; when autograd fires
the hook, the gradient is served to the block. Writing `t.grad = v` swaps a
replacement into that same channel. Because the location is keyed by `id(tensor)`,
gradient errors show a numeric id rather than a module path.

## Gotchas

- **Request `.grad` on the tensor you captured directly** — not on a slice or index
  of it (`hs.grad`, not `hs[0].grad`). An indexing view is a new tensor whose
  gradient isn't the one autograd delivers, and requesting it raises
  `OutOfOrderError`.
- **Access gradients in reverse-forward order.** Requesting an earlier-forward
  tensor's grad before a later one raises `OutOfOrderError`.
- **Only `.grad` is meaningful inside the backward block.** The forward pass is over
  by the time autograd runs; capture any `.output` / `.input` before the
  `with tensor.backward():`.
- **Set `requires_grad_(True)` before backward** if you want a gradient on a
  non-leaf tensor that wouldn't otherwise retain one.
- **A bare `tensor.backward()` is untouched** — it runs vanilla PyTorch and returns
  `None`.

## Related

- [access-and-modify.md](access-and-modify.md) — forward-pass `.output` / `.input`.
- [trace.md](trace.md)
- [save.md](save.md)

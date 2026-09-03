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
import nnsight
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

### Gradients inside a batched invoke

Reading `.grad` on an activation captured in a `tracer.invoke(...)` block works even
when several invokes share the forward — each invoke sees the gradient for *its* rows:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        model.output.logits.save()
    with tracer.invoke("The Great Wall is in"):
        hidden = model.transformer.h[-1].output
        with model.output.logits.sum().backward():
            grad = hidden.grad.save()   # this invoke's rows only
```

The activation an invoke reads is a slice-view of the full batch (not itself in the
loss graph), so nnsight redirects the autograd hook to the full-batch tensor
and recovers this invoke's rows — no extra work on your part. Position indices in a
batch follow the batch's padding, which
[invoke-and-batching.md](invoke-and-batching.md) describes.

The invokes share one forward and therefore one autograd graph, so a second
backward anywhere in the same trace needs `retain_graph=True` on the earlier ones:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        a = model.transformer.h[-1].output
        with model.output.logits.sum().backward(retain_graph=True):
            g_a = a.grad.norm().save()
    with tracer.invoke("The Great Wall is in"):
        b = model.transformer.h[-1].output
        with model.output.logits.sum().backward():
            g_b = b.grad.norm().save()
```

Without it the second one raises
`RuntimeError: Trying to backward through the graph a second time`.

### Read-only gradients without the ordering rule

When you only want to *read* gradients, `retain_grad()` in the forward body plus a
plain `loss.backward()` afterwards gets the same numbers and imposes no ordering at
all — `retain_grad()` is called in forward order, and every `.grad` is available
once the backward finishes:

```python
n_layers = len(model.transformer.h)
with model.trace("Hello world"):
    refs = nnsight.save([])
    for layer in range(n_layers):
        hs = model.transformer.h[layer].output
        hs.retain_grad()
        refs.append(hs)
    loss = model.output.logits.sum().save()

loss.backward()
norms = [ref.grad.norm().item() for ref in refs]
```

The cost is that every `.grad` is materialized and kept. Use `with loss.backward():`
when you want to edit a gradient mid-pass, or to hold only the few you asked for.

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

## Limits

**`model.generate()` produces no gradients.** HuggingFace decorates
`GenerationMixin.generate` with `@torch.no_grad()`, so activations inside a
generation trace come back with `requires_grad=False` and opening a backward block
there raises:

```
NotImplementedError: This tensor does not require grad, so a backward session
cannot produce gradients: nothing the block reads can ever receive one.
```

Wrapping the call in `torch.enable_grad()` does not change this — the decorator is
applied inside `generate` and wins. For gradients over generated tokens, run a loop
of `model.trace` calls over the growing prefix and take the backward in each.

**A frozen model produces no gradients.** After `model.requires_grad_(False)` there
is nothing for autograd to accumulate into, and a backward block raises the same
`NotImplementedError`. Once you inject a tensor that does require grad — a steering
vector, an adapter — gradients exist from that point *downstream* only; reading
`.grad` on an activation upstream of the injection raises
`RuntimeError: cannot register a hook on a tensor that doesn't require gradient`.

**The model's parameter gradients accumulate across traces.** Each backward adds
into `param.grad` the way it does in ordinary PyTorch. Zero them between steps, or
freeze the model and differentiate only what you injected, if you do not want that
memory held. It is the only thing that grows across a long loop: neither the trace
nor `model.session()` retains anything per iteration.

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
- **An activation needs no `requires_grad_(True)`.** A tensor read from `.output`
  or `.input` is already a non-leaf carrying a `grad_fn`, and calling
  `requires_grad_(True)` on it leaves it exactly as it was. The only tensor that
  needs the call is a leaf you construct yourself — a scaled embedding baseline
  for integrated gradients, or a steering vector you are optimizing.
- **A gradient is readable only while the block is open.** Once it closes, `t.grad`
  is `None` again (PyTorch's non-leaf `.grad` warning is the only signal), so
  `.save()` what you want before leaving.
- **A gradient that never arrives raises rather than hangs.** Asking for the
  `.grad` of a tensor autograd never reaches — a branch off the metric's path, for
  instance — surfaces `OutOfOrderError` at the end of the run.
- **A bare `tensor.backward()` is untouched** — it runs vanilla PyTorch and returns
  `None`.

## Related

- [access-and-modify.md](access-and-modify.md) — forward-pass `.output` / `.input`.
- [trace.md](trace.md)
- [save.md](save.md)

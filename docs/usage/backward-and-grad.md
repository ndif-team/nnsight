---
title: Backward and Gradients
one_liner: with tensor.backward(): runs the real backward interleaved so .grad on tensors is readable and editable.
tags: [usage, backward, gradients]
related: [docs/usage/access-and-modify.md, docs/usage/save.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/backward.py, src/nnsight/__init__.py]
---

# Backward and Gradients

`with tensor.backward():` runs the real backward pass **interleaved** with the body
of the `with` block, so the block can read and replace the `.grad` of any tensor as
the gradient reaches it — for gradient attribution, or optimization through a
frozen model. It is almost always nested inside a forward trace, so the tensors
whose gradients you want are the real ones produced during the run. A bare
`tensor.backward()` (no `with`) is vanilla PyTorch and returns `None`.

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

Reading `hs.grad` waits until autograd produces that gradient; assigning
`hs.grad = ...` replaces the gradient that flows onward. Gradient errors name the
tensor by a numeric id rather than a module path.

## Variations

### Read gradients in reverse-forward order

Gradients flow backward through the model, so request `.grad` in the reverse of the
forward order — later layers first; the other order raises `OutOfOrderError`:

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

Every backward but the last needs `retain_graph=True`, including backwards in
different invokes of one trace (they share one forward, so one graph). Without it
the second raises `RuntimeError: Trying to backward through the graph a second time`.

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

nnsight recovers this invoke's rows from the full-batch gradient for you. Position
indices follow the batch's padding ([invoke-and-batching](invoke-and-batching.md)).

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

## Limits

**`model.generate()` produces no gradients.** HuggingFace decorates
`GenerationMixin.generate` with `@torch.no_grad()` (wrapping the call in
`torch.enable_grad()` does not help), so opening a backward block there raises:

```
NotImplementedError: This tensor does not require grad, so a backward session
cannot produce gradients: nothing the block reads can ever receive one.
```

Run a loop of `model.trace` calls over the growing prefix and take the backward in each:

```python
tokens = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
grads = []
for step in range(3):
    with model.trace(tokens):
        hs = model.transformer.h[-1].output
        with model.output.logits[0, -1].max().backward():
            g = hs.grad.norm().save()
        logits = model.output.logits.save()
    grads.append(g)
    tokens = torch.cat([tokens, logits[:, -1].argmax(-1, keepdim=True)], dim=-1)
```

**A frozen model produces no gradients either.** After `model.requires_grad_(False)`
a backward block raises the same `NotImplementedError`. Once you inject a tensor
that requires grad (a steering vector, an adapter), gradients exist downstream of
it only; `.grad` upstream raises
`RuntimeError: cannot register a hook on a tensor that doesn't require gradient`.

## In-place writes after the forward has moved on

Reading a later module's output advances the forward past the earlier one. An
in-place write to the earlier tensor after that point lands on a value the model
has already consumed: on a forward-only trace it silently changes nothing, and
under a backward it trips autograd's version counter:

```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [...] is at version 1; expected version 0 instead.
```

```python
with model.trace("Hello world"):
    hs = model.transformer.h[6].output
    hs *= 2                                 # edit while the forward is still here
    later = model.transformer.h[8].output   # moving `hs *= 2` below this line breaks it
    with model.output.logits.sum().backward():
        grad = later.grad.norm().save()
```

Position breaks a backward, not form: at the interception point `hs *= 2`,
`hs[:] = hs * 2` and `model.transformer.h[6].output = hs * 2` all differentiate.
They differ only in what `hs` names afterwards — the in-place forms alias the
edited value, so `hs.grad` is the gradient at the edit; after the replacement `hs`
is the pre-edit tensor, so its gradient carries the edit's chain rule (here, a
factor of 2). When an intervention seems to do nothing, compare logits with and
without it.

## Gotchas

- **Parameter gradients accumulate across traces**, as in ordinary PyTorch. Zero
  them between steps, or freeze the model and differentiate only what you
  injected. Nothing else grows across a long loop of traces or a session.
- **Request `.grad` on the tensor you captured directly** (`hs.grad`, not
  `hs[0].grad`). An indexing view is a new tensor whose gradient autograd never
  delivers, and requesting it raises `OutOfOrderError`.
- **Only `.grad` is meaningful inside the backward block.** The forward is over by
  then; capture any `.output` / `.input` before `with tensor.backward():`.
- **An activation needs no `requires_grad_(True)`.** A tensor read from `.output`
  or `.input` already carries a `grad_fn`. Only a leaf you construct yourself (an
  integrated-gradients baseline, a steering vector you optimize) needs the call.
- **A gradient is readable only while the block is open.** Afterwards `t.grad` is
  `None` again (PyTorch's non-leaf `.grad` warning is the only signal), so
  `.save()` what you want inside.
- **A gradient that never arrives raises rather than hangs.** `.grad` of a tensor
  off the metric's path surfaces `OutOfOrderError` at the end of the run.
- **A backward block cannot be nested inside another.** It surfaces from inside
  autograd as `ValueError: cyclic parent chain`, naming nothing you wrote. Write
  the two as siblings, `retain_graph=True` on the earlier (plus
  `create_graph=True` when the second differentiates the first's gradient).
- **Interventions inside a gradient-checkpointed segment never reach the
  backward.** `torch.utils.checkpoint` recomputes the segment unmodified for
  autograd, so the forward output is right and the parameter gradients inside the
  segment are silently wrong. Turn checkpointing off while tracing
  (`model.gradient_checkpointing_disable()` on HuggingFace,
  `model.set_grad_checkpointing(False)` on timm).

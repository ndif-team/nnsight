---
title: Backward (Gradient) Pitfalls
one_liner: with tensor.backward() interleaves the real backward — capture forward tensors FIRST, .grad lives on tensors, gradient order is reverse-forward.
tags: [gotcha, backward, grad]
related: [docs/usage/backward-and-grad.md, docs/gotchas/order-and-deadlocks.md]
sources: [src/nnsight/intervention/backward.py, src/nnsight/intervention/interleaver.py:83]
---

# Backward (Gradient) Pitfalls

## TL;DR
- `with tensor.backward():` runs the **real backward pass interleaved** with the block. Capture the forward tensors you want gradients for *before* the backward block, then read `.grad` on them inside it.
- You do **not** need `requires_grad_(True)` — a tensor read from `.output` is already in the autograd graph; reading `.grad` inside the backward block registers a hook on it.
- `.grad` is on **tensors, not modules**. There is no `module.grad`; capture the tensor, then read its `.grad`.
- Gradient access order is the **reverse** of forward access order (gradients flow backward). Requesting an earlier-forward tensor's gradient before a later one raises `OutOfOrderError`.
- `retain_graph=True` on the first backward if you call `.backward()` more than once on overlapping graphs.
- A standalone `with loss.backward():` works outside a forward trace if you `.save()` the forward tensors first.

---

## Read forward values *before* the backward block

### Symptom
Requesting `module.output` inside the backward block raises:
```
OutOfOrderError: 'model.transformer.h.0.output.i0' was requested but the model already ran past it
```

### Cause
The backward block runs interleaved with the *backward* pass, under its own interleaver that only serves `.grad` (`src/nnsight/intervention/backward.py`). The forward pass is already done, so a `.output` request there is never served and surfaces as `OutOfOrderError` at the end of the run.

### Wrong code
```python
with model.trace("Hello world"):
    loss = model.output.logits.sum()
    with loss.backward():
        hs = model.transformer.h[-1].output   # OutOfOrderError — forward is done
        grad = hs.grad.save()
```

### Right code
```python
with model.trace("Hello world"):
    hs = model.transformer.h[-1].output       # capture during the forward
    loss = model.output.logits.sum()
    with loss.backward():
        grad = hs.grad.clone().save()          # read its gradient
```

### Mitigation / how to spot it early
- Treat the backward block as "read/edit gradients of tensors you already hold". Capture any forward intermediate before opening it.

---

## `.grad` is on tensors, not modules

### Cause
Gradients live on tensors. nnsight keys gradient requests on `id(tensor)`, so there is no `module.grad` — capture the tensor from `.output`, then read `.grad` on it.

### Right code
```python
with model.trace("Hello world"):
    hs5 = model.transformer.h[5].output       # the tensor
    loss = model.output.logits.sum()
    with loss.backward():
        g = hs5.grad.clone().save()
```

Editing works too: `hs5.grad = hs5.grad * 2` inside the block replaces the gradient flowing onward (and downstream weight grads reflect it).

---

## Gradient order is the reverse of forward order

### Symptom
```
OutOfOrderError: '140509505995472.grad.i0' was requested but the model already ran past it
```
(The location is `id(tensor).grad` — match the id back to your captured variable.)

### Cause
Backprop reaches the deepest layer first. Requesting `h[0].grad` before `h[10].grad` is the gradient analog of asking for `h[0].output` after `h[10].output` on the forward — the later hook already fired.

### Wrong / Right
```python
with model.trace("Hello world"):
    h0 = model.transformer.h[0].output
    h10 = model.transformer.h[10].output
    loss = model.output.logits.sum()
    with loss.backward():
        g10 = h10.grad.clone().save()   # later-forward gradient first
        g0 = h0.grad.clone().save()     # then earlier
```

---

## `retain_graph=True` for multiple backward passes

### Symptom
`RuntimeError: Trying to backward through the graph a second time ...`.

### Cause / fix
PyTorch frees the graph after the first backward. Pass `retain_graph=True` on all but the last backward.

```python
with model.trace("Hello world"):
    hs = model.transformer.h[-1].output
    logits = model.output.logits
    with logits.sum().backward(retain_graph=True):
        g1 = hs.grad.clone().save()
    with (logits.sum() * 2).backward():
        g2 = hs.grad.clone().save()
```

---

## Standalone backward outside a forward trace

### Cause
`with loss.backward():` is independent of the forward trace; it only needs the loss tensor and the tensors whose `.grad` you read. Save the forward tensors so the graph stays alive.

### Right code
```python
with model.trace("Hello world"):
    hs = model.transformer.h[-1].output.save()
    logits = model.output.logits.save()

with logits.sum().backward():
    grad = hs.grad.clone().save()
print(grad.shape)
```

### Mitigation
- Use this to inspect forward results before deciding to compute gradients, or to keep the forward trace short.

---

## Related
- [docs/usage/backward-and-grad.md](../usage/backward-and-grad.md) — full backward/grad reference.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — the forward analog of the reverse-order rule.

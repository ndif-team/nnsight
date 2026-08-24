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
- All invokes in a trace share **one** autograd graph, so one `.backward()` *per invoke* still counts as more than once — every invoke but the last needs `retain_graph=True`.
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

## One `.backward()` per invoke still counts as more than once

### Symptom
Each invoke calls `.backward()` exactly once, so nothing looks repeated — but the second invoke raises:

```
RuntimeError: The autograd graph for this trace has already been freed.

Every invoke in a trace contributes to a single batched forward pass, so all
invokes share one autograd graph. An earlier `.backward()` in the same trace
freed the graph that this one needs.
```

### Cause
Invokes are not separate runs. Every invoke's input is combined into **one** batch and the model is called **once**, so the whole trace has a single autograd graph. The first `.backward()` frees it and every later invoke walks a graph that no longer exists.

### Wrong code
```python
with model.trace() as tracer:
    with tracer.invoke(prompt_a):
        a1 = model.transformer.h[5].output
        with model.output.logits.sum().backward():          # frees the graph
            grad_a = a1.grad.save()

    with tracer.invoke(prompt_b):
        a1 = model.transformer.h[5].output
        with model.output.logits.sum().backward():          # RuntimeError
            grad_b = a1.grad.save()
```

### Right code
```python
with model.trace() as tracer:
    with tracer.invoke(prompt_a):
        a1 = model.transformer.h[5].output
        with model.output.logits.sum().backward(retain_graph=True):
            grad_a = a1.grad.save()

    with tracer.invoke(prompt_b):
        a1 = model.transformer.h[5].output
        with model.output.logits.sum().backward():          # last one: let it free
            grad_b = a1.grad.save()
```

Each invoke still gets its own gradient — the loss inside an invoke is built from that invoke's rows, so backward only reaches those rows.

### Mitigation / how to spot it early
- Count `.backward()` calls per **trace**, not per invoke. All but the last need `retain_graph=True`.
- If you don't need the gradients together, give each backward pass its own `model.trace()`. That frees the graph between passes and costs less memory than retaining it.

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

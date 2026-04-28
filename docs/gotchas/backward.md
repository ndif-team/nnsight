---
title: Backward (Gradient) Pitfalls
one_liner: Gradient tracing is a separate session — get .output values FIRST, .grad lives on tensors not modules, gradients flow in reverse.
tags: [gotcha, backward, grad]
related: [docs/usage/save.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/tracing/backwards.py:69, src/nnsight/intervention/tracing/backwards.py:81, src/nnsight/__init__.py:154]
---

# Backward (Gradient) Pitfalls

## TL;DR
- `with tensor.backward():` opens a *separate* interleaving session. Get any `.output`/`.input` you need *before* you enter the backward block.
- `.grad` is on tensors, not modules. There's no `module.grad` — request gradients on the tensor you saved during the forward pass.
- Gradient access order is the *reverse* of forward access order. If you accessed `h[5].output` then `h[10].output` on forward, request `.grad` for `h[10]`'s tensor before `h[5]`'s.
- `retain_graph=True` is required if you call `.backward()` more than once on overlapping graphs.
- Standalone `with loss.backward():` outside any `model.trace()` works for simple cases — useful when you save the forward outputs first and want to inspect gradients afterward.

---

## Backward is a separate interleaving session

### Symptom
Inside `with logits.sum().backward():`, you try to access `model.transformer.h[0].output` and get:

```
ValueError: Cannot request `model.transformer.h.0.output` in a backwards tracer.
You can only request `.grad`. Please define your Tensors before the Backwards Tracer
and interact with their gradients within the Backwards Tracer.
```

### Cause
`backward(...)` is hooked at the `torch.Tensor` level (`src/nnsight/__init__.py:154` patches `Tensor.backward`). Entering the backward context creates a fresh `BackwardsMediator` and `Interleaver` (`src/nnsight/intervention/tracing/backwards.py:81`). The `BackwardsMediator` overrides `request` to reject anything that doesn't end in `.grad`:

```python
def request(self, requester):
    if not requester.endswith(".grad"):
        raise ValueError(...)
```

(`src/nnsight/intervention/tracing/backwards.py:69`).

That means inside the backward block you can only access `.grad` on tensors you already captured. The forward-pass `.output`/`.input` machinery is not running there.

### Wrong code
```python
with model.trace("Hello"):
    logits = model.lm_head.output
    with logits.sum().backward():
        # ValueError — can't access .output inside a backward tracer
        hs = model.transformer.h[-1].output
        grad = hs.grad.save()
```

### Right code
```python
with model.trace("Hello"):
    # 1) Capture forward-pass tensors BEFORE the backward block
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    logits = model.lm_head.output

    # 2) Inside the backward block, only access .grad on those captured tensors
    with logits.sum().backward():
        grad = hs.grad.save()
```

### Mitigation / how to spot it early
- Treat the backward block as "read-only on gradients of tensors you already have a handle to".
- If you need a forward intermediate for backward, capture it before opening the backward context.

---

## `.grad` is on tensors, not modules

### Symptom
Errors like `AttributeError: 'Envoy' object has no attribute 'grad'`, or trying `model.transformer.h[5].grad` and seeing nothing useful.

### Cause
Gradients in PyTorch live on tensors via `tensor.grad` and `tensor.register_hook(...)`. nnsight's `wrap_grad` (`src/nnsight/intervention/tracing/backwards.py:10`) hooks into `torch.Tensor.grad` and uses `id(tensor)` as the requester key. There's no `module.grad`; the gradient is on the *tensor* that flows through that module's output.

So you save the tensor on forward, then request `.grad` on it during backward.

### Wrong code
```python
with model.trace("Hello"):
    logits = model.lm_head.output
    with logits.sum().backward():
        # there is no h[5].grad
        g = model.transformer.h[5].grad.save()
```

### Right code
```python
with model.trace("Hello"):
    hs5 = model.transformer.h[5].output
    hs5.requires_grad_(True)
    logits = model.lm_head.output

    with logits.sum().backward():
        g = hs5.grad.save()
```

### Mitigation / how to spot it early
- Module objects don't have gradients; *tensors* do. Capture the tensor first.

---

## Gradient access order is the reverse of forward access order

### Symptom
You saved tensors at multiple layers on the forward pass, then tried to request `.grad` in forward order during the backward block. You get a missed-provider error like `Execution complete but '<id>.grad' was not provided`.

### Cause
Backprop runs in reverse: gradients reach the deepest layer first, then propagate back to the input. The mediator thread inside the backward block synchronizes with `register_hook` callbacks fired in that order. Requesting `h[3].grad` *before* `h[10].grad` is the gradient analog of asking for `h[3].output` after `h[10].output` on forward — the deeper hook has already fired and is past.

### Wrong code
```python
with model.trace("Hello"):
    h3 = model.transformer.h[3].output; h3.requires_grad_(True)
    h10 = model.transformer.h[10].output; h10.requires_grad_(True)
    logits = model.lm_head.output

    with logits.sum().backward():
        g3 = h3.grad.save()    # waits — but h3.grad fires AFTER h10.grad
        g10 = h10.grad.save()  # h10's grad already fired, missed
```

### Right code
```python
with model.trace("Hello"):
    h3 = model.transformer.h[3].output; h3.requires_grad_(True)
    h10 = model.transformer.h[10].output; h10.requires_grad_(True)
    logits = model.lm_head.output

    with logits.sum().backward():
        # reverse of forward order
        g10 = h10.grad.save()
        g3 = h3.grad.save()
```

### Mitigation / how to spot it early
- Mental model: backward reverses the forward order. Mirror your accesses.
- The exception text shows tensor ids like `139820463417744.grad` rather than module paths — that's because `wrap_grad` keys gradients on `id(tensor)`. Match the id back to your captured variable to figure out which one fired in the wrong order.

---

## `retain_graph=True` for multiple backward passes

### Symptom
`RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed.` when you call `.backward()` more than once.

### Cause
PyTorch frees the autograd graph after the first `.backward()` call. nnsight respects this — the second backward sees a freed graph. Pass `retain_graph=True` if you intend to call backward multiple times on overlapping graphs.

### Wrong code
```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output; hs.requires_grad_(True)
    logits = model.lm_head.output

    with logits.sum().backward():
        g1 = hs.grad.save()

    with (logits * 2).sum().backward():    # RuntimeError
        g2 = hs.grad.save()
```

### Right code
```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output; hs.requires_grad_(True)
    logits = model.lm_head.output

    with logits.sum().backward(retain_graph=True):
        g1 = hs.grad.save()

    with (logits * 2).sum().backward():
        g2 = hs.grad.save()
```

### Mitigation / how to spot it early
- If you'll backward twice, the first call needs `retain_graph=True`. The last call doesn't (and skipping it frees memory).

---

## Standalone backward outside a `model.trace()`

### Symptom
You want to inspect gradients of a forward result you already saved, without holding open a forward trace.

### Cause
`BackwardsTracer` is independent of `InterleavingTracer`. It only needs the saved tensor and the loss tensor — no forward trace context required.

### Right code (standalone backward)
```python
# 1) Forward pass — save the tensors you'll want gradients for
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    hs = hs.save()
    logits = model.lm_head.output.save()

# 2) Backward pass — outside the trace
loss = logits.sum()
with loss.backward():
    grad = hs.grad.save()

print(grad.shape)
```

### Mitigation / how to spot it early
- Use this when you want to compute gradients *after* inspecting forward results, or when you want to keep the forward trace context as short as possible.

---

## Related
- [docs/usage/save.md](../usage/save.md) — saving values for later.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — the same mediator/interleaver model applies during backward.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — forward analog of the reverse-order rule.

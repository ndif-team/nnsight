---
title: Backward (Gradient) Pitfalls
one_liner: with tensor.backward() interleaves the real backward — capture forward tensors FIRST, .grad lives on tensors, gradient order is reverse-forward.
tags: [gotcha, backward, grad]
related: [docs/usage/backward-and-grad.md, docs/gotchas/order-and-deadlocks.md]
sources: [src/nnsight/intervention/backward.py, src/nnsight/intervention/interleaver.py]
---

# Backward (Gradient) Pitfalls

## TL;DR
- `with tensor.backward():` runs the **real backward pass interleaved** with the block. Capture the forward tensors you want gradients for *before* the backward block, then read `.grad` on them inside it.
- You do **not** need `requires_grad_(True)` — a tensor read from `.output` is already in the autograd graph; reading `.grad` inside the backward block registers a hook on it.
- `.grad` is on **tensors, not modules**. There is no `module.grad`; capture the tensor, then read its `.grad`.
- Gradient access order is the **reverse** of forward access order (gradients flow backward). Requesting an earlier-forward tensor's gradient before a later one raises `OutOfOrderError`.
- `retain_graph=True` on the first backward if you call `.backward()` more than once on overlapping graphs — including across two invokes of the same trace, which share one graph.
- A standalone `with loss.backward():` works outside a forward trace if you `.save()` the forward tensors first.
- `model.generate()` runs under `torch.no_grad()`, so no backward block works inside one.

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

Two invokes of the same trace count as overlapping: they share one forward pass and
therefore one graph, so a backward in each needs `retain_graph=True` in all but the
last.

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

---

## No gradients inside `model.generate()`

### Symptom
```
NotImplementedError: This tensor does not require grad, so a backward session cannot produce gradients: nothing the block reads can ever receive one.
```

### Cause
HuggingFace decorates `GenerationMixin.generate` with `@torch.no_grad()`. Activations captured inside a generation trace come back with `requires_grad=False`, so there is no graph to back-propagate through. `torch.enable_grad()` around the call does not help — the decorator is applied inside `generate` and takes effect after it.

### Wrong code
```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    for step in tracer.iter[:]:
        hs = model.transformer.h[-1].output
        with model.output.logits.sum().backward():   # NotImplementedError
            grad = hs.grad.save()
```

### Right code
Drive the generation yourself with `model.trace` over the growing prefix, and take a backward per step:

```python
tokens = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
grads = []
for step in range(3):
    with model.trace(tokens):
        hs = model.transformer.h[-1].output
        with model.output.logits[0, -1].max().backward():
            grads.append(hs.grad.norm().save())
        logits = model.output.logits.save()
    tokens = torch.cat([tokens, logits[:, -1].argmax(-1, keepdim=True)], dim=-1)
```

---

## A frozen model has no gradients to give

### Symptom
Same `NotImplementedError` as above, after `model.requires_grad_(False)`. Or, once you have injected a trainable tensor:
```
RuntimeError: cannot register a hook on a tensor that doesn't require gradient
```

### Cause
Freezing removes every parameter from the graph, so nothing an activation touches requires grad. When you then add a tensor that does — a steering vector, an adapter — only activations *downstream* of the injection point carry a gradient. Reading `.grad` on an activation upstream of it is asking autograd for something it never computes.

### Right code
```python
direction = torch.zeros(768, requires_grad=True, device=model.device)

with model.trace("Hello world"):
    model.transformer.h[6].output[:, -1, :] += direction
    downstream = model.transformer.h[9].output        # after the injection
    with model.output.logits.sum().backward():
        grad = downstream.grad.norm().save()
```

---

## In-place writes after the forward has moved on

### Symptom
Either nothing happens, or — as soon as you add a backward:
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1, 10, 768]], which is output 0 of Mul, is at version 1; expected version 0 instead.
```

### Cause
Reading a later module's output advances the forward pass past the earlier one. An in-place write to the earlier tensor after that point lands on a value the model has already consumed: on a forward-only trace it changes nothing, and under a backward it trips autograd's version counter.

### Wrong code
```python
with model.trace("Hello world"):
    hs = model.transformer.h[6].output
    later = model.transformer.h[8].output   # the forward is now past layer 6
    hs *= 2                                 # too late — silently does nothing
    with model.output.logits.sum().backward():
        grad = later.grad.norm().save()     # RuntimeError
```

### Right code
Edit at the point you intercept, before reading anything later:

```python
with model.trace("Hello world"):
    hs = model.transformer.h[6].output
    hs *= 2                                 # while the forward is still here
    later = model.transformer.h[8].output
    with model.output.logits.sum().backward():
        grad = later.grad.norm().save()
```

In-place editing is not itself a problem for autograd. At the interception point
`hs *= 2`, `hs[:] = hs * 2` and `model.transformer.h[6].output = hs * 2` produce the
same forward result and all differentiate. They differ only in what `hs` goes on to
name: the in-place forms leave it aliased to the edited value, so `hs.grad` is the
gradient at the edit; the replacement form leaves `hs` as the tensor from before the
edit, so its gradient carries the edit's own chain rule (here, a factor of 2).
Position is what breaks a backward, not form.

### Mitigation / how to spot it early
- A forward-only trace gives no signal at all, so compare the logits with and without the edit when an intervention seems to do nothing.

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

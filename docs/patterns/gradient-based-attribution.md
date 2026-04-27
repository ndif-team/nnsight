---
title: Gradient-Based Attribution
one_liner: Use `with metric.backward():` inside a trace to compute gradients of intermediate activations w.r.t. a metric - saliency, integrated gradients, layer importance.
tags: [pattern, interpretability, gradients, attribution, saliency]
related: [docs/usage/backward-and-grad.md, docs/patterns/attribution-patching.md, docs/patterns/logit-lens.md]
sources: [src/nnsight/intervention/tracing/tracer.py, src/nnsight/intervention/envoy.py]
---

# Gradient-Based Attribution

## What this is for

Gradient-based attribution methods explain a model's prediction by asking "what changes, in any internal activation, would change the metric most?" The simplest form is the saliency map: `d(metric)/d(activation)`. More structured versions (input * gradient, integrated gradients, GradCAM) reweight or accumulate gradients to satisfy specific axioms.

In nnsight, gradients on intermediate activations are accessed via `with tensor.backward():`. This opens a **separate interleaving session** that exposes `.grad` on tensors captured in the surrounding forward trace. You set `requires_grad_(True)` on the activations you care about, run the metric forward, then enter the backward context and read gradients (in reverse forward order). See `docs/usage/backward-and-grad.md`.

## When to use

- Saliency / sensitivity maps: which positions or features matter most for a prediction.
- Integrated gradients: axiomatic input attribution.
- Layer importance scores from gradient norms.
- The forward+gradient halves of attribution patching - see `docs/patterns/attribution-patching.md`.

## Canonical pattern

Saliency on the residual stream at every layer, last position, for the logit of " Paris":

```python
import torch
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
target = model.tokenizer.encode(" Paris")[0]
n_layers = len(model.transformer.h)

residual_grads = [None] * n_layers

with model.trace(prompt):
    refs = []
    for L in range(n_layers):
        hs = model.transformer.h[L].output
        hs.requires_grad_(True)
        refs.append(hs)

    metric = model.lm_head.output[:, -1, target]

    with metric.sum().backward():
        # Reverse order: backward visits later layers first.
        for L in reversed(range(n_layers)):
            residual_grads[L] = refs[L].grad.save()

# Per-layer saliency: L2 norm of the gradient at the last position.
for L, g in enumerate(residual_grads):
    print(f"layer {L:2d}: ||grad||_2 at last pos = {g[:, -1, :].norm().item():.4f}")
```

**Two key constraints** (see `docs/usage/backward-and-grad.md`):

1. Set `requires_grad_(True)` on every activation you want a gradient on, *before* the metric is computed.
2. Inside `with metric.backward():`, only `.grad` is available - no `.input` / `.output`. Capture the activations themselves outside the backward context.

## Variations

### Input-token saliency (input * grad)

```python
with model.trace(prompt):
    embeds = model.transformer.wte.output
    embeds.requires_grad_(True)
    embeds_save = embeds.save()

    metric = model.lm_head.output[:, -1, target]

    with metric.sum().backward():
        grad = embeds.grad.save()

# Token-level saliency: |embed * grad|, summed over hidden dim.
saliency = (embeds_save * grad).sum(dim=-1).abs()  # [B, S]
tokens = model.tokenizer.convert_ids_to_tokens(model.tokenizer.encode(prompt))
for tok, s in zip(tokens, saliency[0].tolist()):
    print(f"{tok!r:>15}  {s:.3f}")
```

### Integrated gradients (IG)

IG averages gradients along a straight-line path from a baseline embedding (e.g. zeros) to the actual embedding. Run several traces at scaled embeddings:

```python
import torch

baseline = torch.zeros_like  # build at trace time
N_STEPS = 16

ig_accum = None
for step in range(N_STEPS):
    alpha = (step + 0.5) / N_STEPS
    with model.trace(prompt):
        embeds = model.transformer.wte.output
        embeds_full = embeds.save()
        # Replace embeds with alpha * embeds (baseline = 0).
        scaled = embeds * alpha
        scaled.requires_grad_(True)
        model.transformer.wte.output = scaled

        metric = model.lm_head.output[:, -1, target]
        with metric.sum().backward():
            g = scaled.grad.save()

    contribution = embeds_full * g  # input * grad along the path
    ig_accum = contribution if ig_accum is None else ig_accum + contribution

ig = ig_accum / N_STEPS  # [B, S, hidden]
saliency = ig.sum(dim=-1)  # [B, S]
```

For a one-trace remote-friendly version, wrap in a `model.session():` (see `docs/usage/session.md`) and accumulate inside.

### Per-component norm score

Take the L2 norm of `(activation * grad)` to rank components:

```python
score = (act * act.grad).pow(2).sum(dim=-1).mean(dim=(0, 1))   # [hidden]
top = score.topk(10).indices
```

### Modifying gradients during backward

You can write to `.grad` inside the backward context to do gradient surgery (e.g. masking, perturbation):

```python
with model.trace(prompt):
    hs = model.transformer.h[5].output
    hs.requires_grad_(True)
    metric = model.lm_head.output[:, -1, target]
    with metric.sum().backward():
        hs.grad[:, :, 100:200] = 0    # mask a feature band
        captured = hs.grad.save()
```

### Multiple backward passes

Use `retain_graph=True` to backprop more than once:

```python
with model.trace(prompt):
    hs = model.transformer.h[5].output
    hs.requires_grad_(True)
    logits = model.lm_head.output

    with logits[:, -1, target].sum().backward(retain_graph=True):
        g_target = hs.grad.save()

    with logits[:, -1, :].pow(2).sum().backward():
        g_norm = hs.grad.save()
```

## Interpretation tips

- **Saliency norms vs signed gradients answer different questions.** `||grad||` says "how sensitive". `act * grad` says "how much does this feature *currently* contribute".
- **IG axioms (completeness, sensitivity)** make IG more reliable than raw input * grad on saturated networks - but it costs `N_STEPS` forward+backward passes.
- **Gradient saturation** is real. On a confident prediction the gradient at the actual point is small; the path-integral nature of IG fixes this.
- **Position dimension matters.** A gradient summed across positions tells you a layer's importance; a per-position gradient tells you which token.
- **Compare to a randomized baseline** (random labels or shuffled positions) - real attribution should beat it.
- **Use the same metric across runs.** `lm_head.output[..., target]` (the logit) and `softmax(...)[..., target]` (the probability) give numerically very different gradients.

## Gotchas

- `.requires_grad_(True)` must be called *inside* the trace, on the activation tensor as it comes from the model. Setting it on a local variable that aliases the tensor is fine; setting it on a `.clone()` is not.
- Inside `with metric.backward():` you cannot access `.input` / `.output` of any module - only `.grad` of tensors captured in the surrounding forward trace. See `docs/usage/backward-and-grad.md`.
- **Backward access order is reverse of forward.** If you accessed layers 0..N-1 forward, request grads from N-1..0. Out-of-order access can hang or raise.
- Some metrics are not differentiable end-to-end (`argmax`, top-k indices, integer ops). Use logits / log-probs instead.
- For very long sequences, gradients on every layer's residual is memory-heavy. Save only what you need.

## Related

- `docs/usage/backward-and-grad.md` - The full backward-context reference.
- [attribution-patching](attribution-patching.md) - Combines clean activations with corrupt-run gradients.
- [logit-lens](logit-lens.md) - Often used together: which layer first cares (gradient) about which prediction (logit lens).
- [steering](steering.md) - One way to validate a gradient-discovered direction is to add it back into the residual.

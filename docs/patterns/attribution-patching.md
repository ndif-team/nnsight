---
title: Attribution Patching
one_liner: Linear approximation of activation patching - one clean forward, one corrupt forward+backward, then `(act_clean - act_corrupt) * grad_corrupt` per component.
tags: [pattern, interpretability, gradients, attribution, patching]
related: [docs/usage/backward-and-grad.md, docs/patterns/activation-patching.md, docs/patterns/gradient-based-attribution.md]
sources: [src/nnsight/intervention/tracing/tracer.py, src/nnsight/intervention/envoy.py]
---

# Attribution Patching

## What this is for

Attribution patching (Nanda, 2023) is a fast linear approximation of activation patching. Where full patching costs one forward pass *per component you want to test*, attribution patching gets you a saliency map over **every component at every position** from a single clean forward and a single corrupt forward+backward.

The approximation is a first-order Taylor expansion. For a component activation `a` and a metric `M(a)`:

```
M(a_clean) - M(a_corrupt) ≈ (a_clean - a_corrupt) · grad_a M | a = a_corrupt
```

So the per-component "patching effect" of swapping clean→corrupt at component `c` is approximated by the elementwise product of `(act_clean - act_corrupt)` and the gradient of the corrupt-run metric with respect to that activation, summed over the component's dimensions.

In nnsight you compute this with two traces (or one session): a clean trace to grab activations, then a corrupt trace that uses `with metric.backward():` to expose `.grad` on the activations of interest. See `docs/usage/backward-and-grad.md`.

Tutorial mirror: https://nnsight.net/notebooks/tutorials/attribution_patching/

## When to use

- Building a per-(layer, position) attribution map without paying `O(layers * positions)` forward passes.
- First-pass screening: use attribution patching to pick the top-K components, then verify with full activation patching on those.
- Circuit-level attribution: you want a heatmap of all components, not a single test.

## Canonical pattern

Logit-difference metric on a clean / corrupt prompt pair (toy example using GPT-2). The metric is `logit[answer_clean] - logit[answer_corrupt]` evaluated on the corrupt run.

```python
import torch
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

clean   = "The Eiffel Tower is in the city of"   # answer: " Paris"
corrupt = "The Colosseum is in the city of"      # answer: " Rome"

paris = model.tokenizer.encode(" Paris")[0]
rome  = model.tokenizer.encode(" Rome")[0]

n_layers = len(model.transformer.h)

# Pass 1: clean activations at every layer's residual output.
clean_acts = [None] * n_layers
with model.trace(clean):
    for L in range(n_layers):
        clean_acts[L] = model.transformer.h[L].output[0].save()

# Pass 2: corrupt forward + backward; capture corrupt activations and grads.
corrupt_acts = [None] * n_layers
corrupt_grads = [None] * n_layers

with model.trace(corrupt):
    hidden_refs = []
    for L in range(n_layers):
        hs = model.transformer.h[L].output[0]
        hs.requires_grad_(True)
        hidden_refs.append(hs)
        corrupt_acts[L] = hs.save()

    logits = model.lm_head.output[:, -1, :]
    metric = logits[:, paris] - logits[:, rome]      # we want to maximize

    with metric.sum().backward():
        for L, hs in enumerate(hidden_refs):
            corrupt_grads[L] = hs.grad.save()

# Compute attribution per layer (sum over batch, seq, hidden).
attribution = torch.tensor([
    ((clean_acts[L] - corrupt_acts[L]) * corrupt_grads[L]).sum().item()
    for L in range(n_layers)
])

for L, a in enumerate(attribution.tolist()):
    print(f"layer {L:2d}: attribution = {a:+.4f}")
```

A high positive score at layer L means "swapping the corrupt residual for the clean one at layer L would significantly raise the (Paris - Rome) logit gap" - i.e. layer L's residual carries city-relevant information.

**Order rules:**

- Inside the corrupt trace, modules are accessed in forward order (`L = 0, 1, ..., n-1`).
- Inside the `metric.sum().backward():` context, gradients should be accessed in reverse order. The example uses `for L, hs in enumerate(hidden_refs)` for clarity; if you hit a "value was not provided" error, reverse the loop. See `docs/usage/backward-and-grad.md`.

## Variations

### Per-position attribution (heatmap)

Drop the `.sum()` over the seq dimension to get a `[layer, seq]` heatmap:

```python
heatmap = torch.stack([
    ((clean_acts[L] - corrupt_acts[L]) * corrupt_grads[L]).sum(dim=-1).squeeze(0)
    for L in range(n_layers)
], dim=0)   # [n_layers, seq]
```

### Sub-block attribution (attention vs MLP)

Track `block.attn.output[0]` and `block.mlp.output` instead of (or in addition to) the residual.

### Per-head attribution

Reshape attention output into `[B, S, n_heads, head_dim]` and do the elementwise product per-head, summing only over `head_dim`. See `docs/patterns/per-head-attention.md`.

### Tracing in one session

To run both passes as a single remote request:

```python
with model.session(remote=True):
    with model.trace(clean):
        for L in range(n_layers):
            clean_acts[L] = model.transformer.h[L].output[0]   # no .save()
    with model.trace(corrupt):
        # ...same as above...
```

In a session, intermediate values cross trace boundaries without `.save()`. See `docs/usage/session.md`.

## Interpretation tips

- **Attribution can have either sign.** Positive = clean is better than corrupt at this component. Negative = the corrupt activation is actually more aligned with the metric here.
- **Magnitude is comparable across components only when summing the same number of dimensions.** Per-layer sums of a residual are comparable across layers; comparing residual sums to per-head sums is not.
- **First-order approximation has limits.** Attribution patching is exact when the metric is a linear function of the activation. For deep networks it is a useful screen, not ground truth - validate top components with full activation patching.
- **Effect of a component**: many practitioners compute `attribution[L] / |full_metric_diff|` to get a fractional contribution.
- **Normalize per-row when plotting heatmaps**, otherwise one large layer drowns out the rest.

## Gotchas

- You must `requires_grad_(True)` on every activation you want a gradient on, *before* you compute the metric.
- Inside `with metric.backward():`, only `.grad` is accessible - no `.input` / `.output`. Capture the activations themselves *before* entering the backward context. See `docs/usage/backward-and-grad.md`.
- Forward order vs backward order: forward access in execution order, backward access in reverse. Skipping this can produce silent errors or hangs.
- Different attention implementations expose different op trees - see `docs/patterns/attention-patterns.md`.
- The metric must be differentiable end-to-end. Argmax, top-k indices etc. are not.

## Related

- [activation-patching](activation-patching.md) - The exact (and slower) operation that attribution patching approximates.
- [gradient-based-attribution](gradient-based-attribution.md)
- `docs/usage/backward-and-grad.md`
- https://nnsight.net/notebooks/tutorials/attribution_patching/
- Nanda (2023), "Attribution Patching: Activation Patching at Industrial Scale".

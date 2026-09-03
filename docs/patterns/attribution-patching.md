---
title: Attribution Patching
one_liner: Linear approximation of activation patching - one clean forward, one corrupt forward+backward, then `(act_clean - act_corrupt) * grad_corrupt` per component.
tags: [pattern, interpretability, gradients, attribution, patching]
related: [docs/usage/backward-and-grad.md, docs/patterns/activation-patching.md, docs/patterns/gradient-based-attribution.md]
sources: [src/nnsight/intervention/backward.py, src/nnsight/intervention/envoy.py]
---

# Attribution Patching

## What this is for

Attribution patching (Nanda, 2023) is a fast linear approximation of activation
patching. Where full patching costs one forward pass *per component you test*,
attribution patching gets a saliency map over **every component at every position**
from a single clean forward and a single corrupt forward+backward.

The approximation is a first-order Taylor expansion. For a component activation `a`
and metric `M(a)`:

```
M(a_clean) - M(a_corrupt) ≈ (a_clean - a_corrupt) · grad_a M | a = a_corrupt
```

So the per-component "patching effect" of swapping corrupt→clean at component `c` is
the elementwise product of `(act_clean - act_corrupt)` and the gradient of the
corrupt-run metric w.r.t. that activation, summed over the component's dimensions.

In nnsight: a clean trace to grab activations, then a corrupt trace whose
`with metric.backward():` exposes `.grad`. See `docs/usage/backward-and-grad.md`.

Tutorial mirror: https://nnsight.net/notebooks/tutorials/attribution_patching/

## When to use

- A per-(layer, position) attribution map without `O(layers × positions)` forwards.
- First-pass screening: pick top-K components with attribution patching, then verify
  the survivors with full activation patching.
- Circuit-level attribution: a heatmap of all components, not a single test.

## Canonical pattern

Logit-difference metric on a clean / corrupt prompt pair. The metric is
`logit[Paris] - logit[Rome]`, evaluated on the corrupt run.

```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

clean   = "The Eiffel Tower is in the city of"   # answer: " Paris"
corrupt = "The Colosseum is in the city of"      # answer: " Rome"
paris = model.tokenizer.encode(" Paris")[0]
rome  = model.tokenizer.encode(" Rome")[0]
n_layers = len(model.transformer.h)

# Pass 1: clean activations at every layer's residual output (no grad needed).
clean_acts = [None] * n_layers
with torch.no_grad():
    with model.trace(clean):
        for L in range(n_layers):
            clean_acts[L] = model.transformer.h[L].output.save()

# Pass 2: corrupt forward + backward; capture corrupt acts and grads.
corrupt_acts = [None] * n_layers
corrupt_grads = [None] * n_layers
with model.trace(corrupt):
    refs = []
    for L in range(n_layers):
        hs = model.transformer.h[L].output
        refs.append(hs)
        corrupt_acts[L] = hs.save()
    logits = model.lm_head.output[:, -1, :]
    metric = logits[:, paris] - logits[:, rome]
    with metric.sum().backward():
        for L in reversed(range(n_layers)):        # reverse-forward order
            corrupt_grads[L] = refs[L].grad.save()

attribution = torch.tensor([
    ((clean_acts[L] - corrupt_acts[L]) * corrupt_grads[L]).sum().item()
    for L in range(n_layers)
])
for L, a in enumerate(attribution.tolist()):
    print(f"layer {L:2d}: attribution = {a:+.4f}")
```

```
layer  0: attribution = -0.4535
layer  1: attribution = -1.3249
layer  2: attribution = +0.2888
...
layer 10: attribution = +2.7499
layer 11: attribution = +3.1342
```

A high positive score at layer L means "swapping the corrupt residual for the clean
one at layer L would raise the (Paris − Rome) logit gap" — i.e. layer L's residual
carries city-relevant information.

**Order rules:**

- In the corrupt trace, access modules in forward order (`L = 0 … n-1`).
- Inside `with metric.sum().backward():`, request gradients in **reverse** order
  (`reversed(range(n_layers))`); forward order there raises `OutOfOrderError`.
- Intermediate activations (layer outputs) are already in the graph, so no
  `requires_grad_(True)` is needed. Run the corrupt pass *without* `torch.no_grad()`.

## Variations

### Per-position attribution (heatmap)

Drop the `.sum()` over the seq dimension for a `[layer, seq]` heatmap:

```python
heatmap = torch.stack([
    ((clean_acts[L] - corrupt_acts[L]) * corrupt_grads[L]).sum(dim=-1).squeeze(0)
    for L in range(n_layers)
], dim=0)   # torch.Size([12, 10]) for this prompt
```

A column index is a position in the batch's padded sequence, not in the prompt —
see [invoke-and-batching.md](../usage/invoke-and-batching.md). Pairs whose clean and
corrupt prompts tokenize to different lengths do not line up column by column;
assert equal token counts, or index from the end.

### Sub-block attribution (attention vs MLP)

Track `block.attn.output[0]` (attention output is a **tuple** — index `[0]` for the
hidden tensor) and `block.mlp.output` (a plain tensor) instead of, or in addition
to, the block residual.

### Per-head attribution

Reshape attention output into `[B, S, n_heads, head_dim]` and take the elementwise
product per head, summing only over `head_dim`. See
[per-head-attention](per-head-attention.md).

### Both passes in one session

To ship both passes as a single (remote) request, wrap them in
`with model.session():` — a value read in one trace flows into the next without an
explicit `.save()`:

```python
with model.session():
    with model.trace(clean):
        for L in range(n_layers):
            clean_acts[L] = model.transformer.h[L].output   # no .save() inside a session
    with model.trace(corrupt):
        ...   # same as above
```

See `docs/usage/session.md`.

## Interpretation tips

- **Attribution has a sign.** Positive = clean beats corrupt here; negative = the
  corrupt activation is more aligned with the metric.
- **Magnitudes compare only across equal-dimension sums.** Per-layer residual sums
  are comparable across layers; residual sums vs per-head sums are not.
- **First-order limits.** Exact only when the metric is linear in the activation;
  for deep nets it's a screen, not ground truth — validate top components with full
  activation patching.
- **Validate against a local patch, not a whole-layer one.** Overwriting layer L's
  entire residual output with the clean one makes the rest of the run a
  deterministic function of a clean state, so every layer returns the same metric —
  on this prompt pair, `2.4247` at all twelve layers, spread `1.2e-05`. Correlating
  attribution against that vector correlates it with float32 rounding. Patch a
  position span, a head, or a sub-block, where what happens after layer L still
  depends on layer L.
- **Fractional effect:** many practitioners report `attribution[L] / |full_diff|`.
- **Normalize per-row** when plotting heatmaps, or one big layer drowns the rest.

## Gotchas

- **Request `.grad` in reverse-forward order** inside the backward context.
- **Only `.grad` inside `with metric.backward():`** — capture activations in the
  forward body first.
- **Attention output is a tuple** (`.output[0]` is the hidden tensor); a block's and
  the MLP's `.output` are plain tensors. See [attention-patterns](attention-patterns.md).
- **The metric must be differentiable** end-to-end.

## Related

- [activation-patching](activation-patching.md) — the exact (slower) operation this approximates.
- [gradient-based-attribution](gradient-based-attribution.md)
- `docs/usage/backward-and-grad.md`
- https://nnsight.net/notebooks/tutorials/attribution_patching/
- Nanda (2023), "Attribution Patching: Activation Patching at Industrial Scale".

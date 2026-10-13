---
title: Gradient-Based Attribution
one_liner: Use `with metric.backward():` inside a trace to read `.grad` on intermediate activations - saliency, integrated gradients, layer importance.
tags: [pattern, interpretability, gradients, attribution, saliency]
related: [docs/usage/backward-and-grad.md, docs/patterns/attribution-patching.md, docs/patterns/logit-lens.md]
sources: [src/nnsight/intervention/backward.py, src/nnsight/intervention/envoy.py]
---

# Gradient-Based Attribution

## What this is for

Gradient-based attribution explains a prediction by asking "what change in any
internal activation would move the metric most?" The simplest form is the saliency
map, `d(metric)/d(activation)`. Structured versions (input × gradient, integrated
gradients) reweight or accumulate gradients to satisfy specific axioms.

In nnsight, `with metric.backward():` runs the real backward pass interleaved with
the block and exposes `.grad` on any tensor captured in the forward trace. Use it for
saliency maps, integrated gradients, and gradient-norm layer importance; for
clean-minus-corrupt × gradient maps see [attribution-patching](attribution-patching.md).

## Canonical pattern

Per-layer residual saliency at the last position, for the logit of " Paris":

```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
target = model.tokenizer.encode(" Paris")[0]
n_layers = len(model.transformer.h)

residual_grads = [None] * n_layers
with model.trace(prompt):
    refs = [model.transformer.h[L].output for L in range(n_layers)]
    metric = model.lm_head.output[:, -1, target]
    with metric.sum().backward():
        for L in reversed(range(n_layers)):     # reverse-forward order
            residual_grads[L] = refs[L].grad.save()

for L, g in enumerate(residual_grads):
    print(f"layer {L:2d}: ||grad|| last pos = {g[:, -1, :].norm().item():.4f}")
```

```
layer  0: ||grad|| last pos = 2.6912
layer  1: ||grad|| last pos = 4.0688
...
layer 10: ||grad|| last pos = 2.8377
layer 11: ||grad|| last pos = 0.6846
```

Request `.grad` in **reverse** forward order, capture the activations before
entering the backward context, and run without `torch.no_grad()` — see
[backward-and-grad](../usage/backward-and-grad.md). In-graph activations (layer,
attention, embedding outputs) need no `requires_grad_(True)`; only a leaf tensor you
construct yourself does.

## Variations

### Input-token saliency (input × grad)

```python
with model.trace(prompt):
    embeds = model.transformer.wte.output
    embeds_save = embeds.save()
    metric = model.lm_head.output[:, -1, target]
    with metric.sum().backward():
        grad = embeds.grad.save()

saliency = (embeds_save * grad).sum(dim=-1).abs()[0]   # [S]
tokens = model.tokenizer.convert_ids_to_tokens(model.tokenizer.encode(prompt))
for tok, s in zip(tokens, saliency.tolist()):
    print(f"{tok!r:>12}  {s:.3f}")
```

The final `'Ġof'` (9.5) and `'Ġis'` (5.5) dominate; the subject tokens score 0.5–1.9.

### Integrated gradients (IG)

IG averages gradients along a straight-line path from a baseline to the actual
activation. Two things decide whether the result means anything, and both are easy
to get wrong:

!!! warning "Pick the baseline deliberately"

    **Scaling `wte.output` alone is not a zero baseline.** GPT-2's transformer is
    `wte -> wpe -> drop -> h`, so the positional embedding is added *after* `wte`:
    at `alpha=0` the model still sees full positional information. Scale
    `drop.output` (or block 0's input) if you want the embedding path to actually
    start from zero.

    **A zero baseline diverges on a residual stream.** LayerNorm is scale
    invariant, so its Jacobian grows like `1/alpha` as the activation shrinks —
    the first sample sits on a singularity and the integral does not converge.
    Measured on GPT-2 layer 6, `sum(IG)` against a target of `+6.29`:
    `N=64 -> -237`, `N=128 -> -77`, `N=256 -> +1107`, `N=512 -> -64`. More steps
    make it worse. Start the path at `alpha = 0.05` instead — 1.30% completeness
    error at `N=64`, 0.36% at `N=128`, 0.09% at `N=256` — or use a mean /
    corrupted-run baseline. A path that starts above zero covers only
    `[alpha_start, 1]`, so scale the accumulated attribution by
    `(1 - alpha_start)`; without that factor completeness is off by exactly that
    much.

The scaled activation is a leaf you construct, so it needs `requires_grad_(True)`:

```python
ALPHA_START = 0.05              # not 0 — see the warning above
N_STEPS = 128
ig_accum = None
for step in range(N_STEPS):
    alpha = ALPHA_START + (1 - ALPHA_START) * (step + 0.5) / N_STEPS
    with model.trace(prompt):
        embeds = model.transformer.wte.output
        embeds_full = embeds.save()
        scaled = (embeds * alpha).detach()
        scaled.requires_grad_(True)                 # constructed leaf
        model.transformer.wte.output = scaled
        metric = model.lm_head.output[:, -1, target]
        with metric.sum().backward():
            g = scaled.grad.save()
    contribution = embeds_full * g
    ig_accum = contribution if ig_accum is None else ig_accum + contribution

ig = (1 - ALPHA_START) * ig_accum / N_STEPS   # path length x mean gradient
saliency = ig.sum(dim=-1)                     # [B, S]
```

**Always check completeness.** The attributions must sum to the change in the
metric across the path; this is the only thing that distinguishes a converged run
from the divergent one above:

```python
def F_at(alpha):                       # scale whatever the IG loop scaled
    with model.trace(prompt):
        embeds = model.transformer.wte.output
        model.transformer.wte.output = embeds * alpha
        value = model.lm_head.output[:, -1, target].save()
    return value                       # return after the block, not inside it

total = ig.sum().item()
target_delta = F_at(1.0).item() - F_at(ALPHA_START).item()
print(f"completeness: {total:.4f} vs {target_delta:.4f}")   # want these to match
```

`F_at` has to scale the same activation the loop scaled, at the same module — a
completeness check against a different module measures nothing.

**Batch the steps.** The alpha steps don't interact, so run 16 at a time as rows of
one batch; `.sum().backward()` gives per-row gradients. This replaces the loop above,
scales the same `wte.output`, and is an order of magnitude faster on GPT-2:

```python
alphas = ALPHA_START + (1 - ALPHA_START) * (torch.arange(N_STEPS) + 0.5) / N_STEPS
ig_accum = 0
for start in range(0, N_STEPS, 16):
    chunk = alphas[start : start + 16]
    with model.trace([prompt] * len(chunk)):
        embeds = model.transformer.wte.output          # rows identical
        embeds_full = embeds[:1].save()
        scaled = (embeds * chunk.view(-1, 1, 1).to(embeds.device)).detach()
        scaled.requires_grad_(True)
        model.transformer.wte.output = scaled
        with model.lm_head.output[:, -1, target].sum().backward():
            g = scaled.grad.sum(0, keepdim=True).save()
    ig_accum = ig_accum + embeds_full * g

ig = (1 - ALPHA_START) * ig_accum / N_STEPS
```

To run all steps as one (remote-friendly) request, wrap the loop in
`with model.session():` — see [session](../usage/session.md).

### Editing gradients mid-backward

Assigning `t.grad = ...` inside the backward context replaces the gradient that
flows onward — gradient surgery (masking, perturbation):

```python
with model.trace(prompt):
    hs = model.transformer.h[5].output
    metric = model.lm_head.output[:, -1, target]
    with metric.sum().backward():
        original = hs.grad.clone().save()
        hs.grad = hs.grad * 2       # doubled from here down the graph
        doubled = hs.grad.save()
# torch.equal(original * 2, doubled) -> True
```

To backprop more than once from the same forward, pass `retain_graph=True` — see
[backward-and-grad](../usage/backward-and-grad.md).

## Interpretation tips

- **`||grad||` vs `act * grad`** answer different questions: sensitivity vs current
  contribution.
- **Gradient saturation:** on a confident prediction the gradient at the actual
  point is small; IG's path integral fixes this, at `N_STEPS`× the cost.
- **Position matters.** Summing across positions ranks layers; per-position
  gradients tell you which token.
- **Same metric across runs.** The logit `lm_head.output[..., target]` and the
  probability `softmax(...)[..., target]` give very different gradients.

## Gotchas

- **Non-differentiable metrics** (`argmax`, top-k indices, integer ops) have no
  usable gradient — use logits / log-probs.
- **Memory:** gradients on every layer's residual for a long sequence are heavy.
  Save only what you need.

## Related

- [backward-and-grad](../usage/backward-and-grad.md) — the full backward-context reference.
- [attribution-patching](attribution-patching.md) — clean activations × corrupt-run gradients.
- [logit-lens](logit-lens.md) — which layer first cares (gradient) about which prediction (logit lens).

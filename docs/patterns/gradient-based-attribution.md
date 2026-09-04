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

In nnsight, gradients on intermediate activations are read via
`with tensor.backward():`. This runs the real backward pass **interleaved** with the
block, exposing `.grad` on any tensor you captured in the surrounding forward trace
as the gradient reaches it. See `docs/usage/backward-and-grad.md`.

## When to use

- Saliency / sensitivity maps: which positions or features matter most.
- Integrated gradients: axiomatic input attribution.
- Layer importance scores from gradient norms.
- The forward+gradient halves of attribution patching — see
  [attribution-patching](attribution-patching.md).

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

**Two rules:**

1. **Request `.grad` in reverse-forward order.** Backward visits later layers
   first; asking for an earlier layer's gradient before a later one raises
   `OutOfOrderError`.
2. **Capture the activations you want gradients for *before* entering the backward
   context**, and read their `.grad` *inside* it.

> **No `requires_grad_(True)` needed for in-graph activations.** A layer output,
> attention output, or embedding output is already a non-leaf tensor carrying a
> `grad_fn`, so `with metric.backward():` reads its gradient directly (the examples
> above have no `requires_grad_` and work). Calling `.requires_grad_(True)` on such
> a tensor is harmless but redundant. You only need it for a leaf tensor you
> construct yourself (e.g. a scaled embedding baseline in integrated gradients).
> Run *without* `torch.no_grad()` — the metric must be able to build a graph.

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

```
       'The'  1.938
        'ĠE'  1.789
       'iff'  1.847
        'el'  0.528
    'ĠTower'  1.215
       'Ġis'  5.466
       'Ġin'  2.676
      'Ġthe'  1.567
     'Ġcity'  1.934
       'Ġof'  9.503
```

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

Here the scaled activation *is* a leaf you construct, so it needs
`requires_grad_(True)`:

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

`(1 - ALPHA_START)` is the length of the path being integrated. Drop it and every
attribution is inflated by `1 / (1 - ALPHA_START)`, which the completeness check
below will report as a constant relative error.

**Always check completeness.** IG's defining axiom is that the attributions sum to
the change in the metric across the path. If they don't, the result is an artifact —
this is the only thing that distinguishes a converged run from the divergent one
above, and it costs three lines:

```python
def F_at(alpha):                       # scale whatever the IG loop scaled
    with model.trace(prompt):
        embeds = model.transformer.wte.output
        model.transformer.wte.output = embeds * alpha
        value = model.lm_head.output[:, -1, target].save()
    return value          # `return` *inside* the block is a SyntaxError -- the
                          # body is recompiled at module level, not as a function

total = ig.sum().item()
target_delta = F_at(1.0).item() - F_at(ALPHA_START).item()
print(f"completeness: {total:.4f} vs {target_delta:.4f}")   # want these to match
```

`F_at` has to scale the same activation the loop scaled, at the same module — a
completeness check against a different module measures nothing.

**Batch the steps.** The alpha steps don't interact, so put several in one batch and
let `metric.sum().backward()` give you per-row gradients — 14x faster per step than
one trace per alpha on GPT-2, for a few hundred MiB:

```python
alphas = ALPHA_START + (1 - ALPHA_START) * (torch.arange(N_STEPS) + 0.5) / N_STEPS
chunk = alphas[s : s + 16]                       # 16 is a reasonable knee
with model.trace([prompt] * len(chunk)):
    act = model.transformer.h[LAYER].output      # rows identical
    scaled = (act * chunk.view(-1, 1, 1)).detach()
    scaled.requires_grad_(True)
    model.transformer.h[LAYER].output = scaled
    with model.lm_head.output[:, -1, target].sum().backward():
        g = scaled.grad.sum(0, keepdim=True).save()
```

Replacement with a detached leaf (above), replacement with a non-leaf
(`scaled = act * alpha`, no `detach`), and in-place (`act[:] = act * alpha`, then
read `act.grad`) all give the same gradient — pick whichever reads best.

To run all `N_STEPS` as one (remote-friendly) request, wrap in
`with model.session():` and accumulate inside — see `docs/usage/session.md`.

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

### Multiple backward passes

Use `retain_graph=True` to backprop more than once from the same forward:

```python
with model.trace(prompt):
    hs = model.transformer.h[5].output
    logits = model.lm_head.output
    with logits[:, -1, target].sum().backward(retain_graph=True):
        g_target = hs.grad.norm().save()
    with logits[:, -1, :].pow(2).sum().backward():
        g_norm = hs.grad.norm().save()
```

## Interpretation tips

- **`||grad||` vs `act * grad`** answer different questions: sensitivity vs current
  contribution.
- **IG axioms** (completeness, sensitivity) make IG more reliable than raw
  input × grad on saturated networks — at `N_STEPS`× the cost.
- **Gradient saturation** is real: on a confident prediction the gradient at the
  actual point is small; IG's path integral fixes this.
- **Position matters.** Summing across positions ranks layers; per-position
  gradients tell you which token.
- **Same metric across runs.** The logit `lm_head.output[..., target]` and the
  probability `softmax(...)[..., target]` give very different gradients.

## Gotchas

- **Reverse-forward order for `.grad`.** Requesting an earlier-forward tensor's
  gradient before a later one raises `OutOfOrderError`.
- **Only `.grad` inside `with metric.backward():`** — no `.input` / `.output`.
  Capture activations in the forward body, read their gradients in the backward body.
- **Run without `torch.no_grad()`.** The forward must build a graph for the backward
  to traverse.
- **Non-differentiable metrics** (`argmax`, top-k indices, integer ops) have no
  usable gradient — use logits / log-probs.
- **Memory:** gradients on every layer's residual for a long sequence are heavy.
  Save only what you need.

## Related

- `docs/usage/backward-and-grad.md` — the full backward-context reference.
- [attribution-patching](attribution-patching.md) — clean activations × corrupt-run gradients.
- [logit-lens](logit-lens.md) — which layer first cares (gradient) about which prediction (logit lens).

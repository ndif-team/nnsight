---
title: Steering with Added Vectors
one_liner: Add a precomputed direction to the residual stream at a specific layer to push the model's behavior in a target direction.
tags: [pattern, interpretability, steering, residual-stream]
related: [docs/usage/access-and-modify.md, docs/usage/session.md, docs/patterns/ablation.md, docs/patterns/multi-prompt-comparison.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracing/tracer.py]
---

# Steering with Added Vectors

## What this is for

Activation steering (a.k.a. "activation addition", ActAdd) modifies model behavior by adding a fixed vector to a specific layer's residual stream during the forward pass. The vector is usually a "concept direction" - the difference of mean activations between contrasting prompt sets - or a learned probe direction. Adding it pushes the residual stream toward that concept; subtracting suppresses it.

The same machinery is used for refusal-direction work, sentiment / topic steering, sycophancy mitigation, and "control vectors". The interpretability claim is: if a single low-rank addition to one layer reliably changes behavior, that behavior has a linearly-decodable representation at that layer.

In nnsight this is just an in-place `+=` to `block.output[0]` inside a trace, plus a separate (typically session-level) computation of the steering direction.

## When to use

- Testing whether a behavior is linearly steerable from a given layer.
- Comparing steering effectiveness across layers / coefficients.
- Building refusal-direction / sycophancy / persona interventions.
- Measuring the "direction" learned by a probe by re-injecting it into the model.

## Canonical pattern

Add a random direction at one layer (replace `direction` with your real vector):

```python
import torch
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

LAYER = 6
hidden = model.config.n_embd

# Replace with a real direction from a contrast set.
direction = torch.randn(hidden, device=model.device)
direction = direction / direction.norm()
coef = 4.0

prompt = "I went to the bakery and"

with model.trace() as tracer:
    with tracer.invoke(prompt):
        baseline = model.lm_head.output[:, -1, :].save()

    with tracer.invoke(prompt):
        model.transformer.h[LAYER].output[0][:, -1, :] += direction * coef
        steered = model.lm_head.output[:, -1, :].save()

print("baseline argmax:", model.tokenizer.decode(baseline.argmax(-1)[0]))
print("steered  argmax:", model.tokenizer.decode(steered.argmax(-1)[0]))
```

Two invokes write to disjoint output variables and only the second writes to the steered module's output, so **no barrier is needed**.

## Computing the steering direction

The most common recipe: take the difference of mean residuals between a positive and negative prompt set at the chosen layer.

```python
import torch
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

positive = ["I love this so much", "This is wonderful", "I am very happy"]
negative = ["I hate this so much", "This is terrible", "I am very sad"]

LAYER = 6

# Bundle into one session (one engine call if remote).
with model.session():
    pos_acts = []
    neg_acts = []

    with model.trace() as tracer:
        for p in positive:
            with tracer.invoke(p):
                pos_acts.append(model.transformer.h[LAYER].output[0][:, -1, :].save())
        for p in negative:
            with tracer.invoke(p):
                neg_acts.append(model.transformer.h[LAYER].output[0][:, -1, :].save())

pos = torch.cat([a for a in pos_acts], dim=0).mean(0)   # [hidden]
neg = torch.cat([a for a in neg_acts], dim=0).mean(0)
direction = (pos - neg)
direction = direction / direction.norm()
```

Sweep `coef` and the layer to find the operating point. Typical magnitudes are a few units of the residual norm; very large coefficients (`coef > 10`) usually break fluency.

## Variations

### Steer all positions vs only the last

```python
# All positions: nudges every token's residual.
model.transformer.h[LAYER].output[0][:] += direction * coef

# Only the last position: cleaner targeting on next-token prediction.
model.transformer.h[LAYER].output[0][:, -1, :] += direction * coef

# A specific span (e.g. positions 5-10): targeted intervention.
model.transformer.h[LAYER].output[0][:, 5:10, :] += direction * coef
```

### Multi-layer steering

Adding the same direction at multiple consecutive layers often produces stronger and cleaner effects than a single big push at one layer.

```python
with model.trace(prompt):
    for L in [4, 5, 6, 7]:
        model.transformer.h[L].output[0][:, -1, :] += direction * (coef / 4)
    out = model.lm_head.output.save()
```

### Steering during generation

Wrap the forward intervention with `model.generate(...)` and the addition will fire on every generation step (the trace body runs once per step):

```python
with model.generate(prompt, max_new_tokens=20) as tracer:
    model.transformer.h[LAYER].output[0][:, -1, :] += direction * coef
    text = tracer.result.save()

print(model.tokenizer.decode(text[0]))
```

For step-conditional steering (e.g. steer only on the first 5 tokens), use `tracer.iter[...]`. See `docs/usage/iter.md`.

### Refusal direction

The "refusal direction" line of work computes `mean(harmful prompts) - mean(harmless prompts)` at a middle layer's residual and *subtracts* it (or zero-projects it out) to suppress refusal. The pattern is the same as above with `coef < 0` or with projection (`hs - (hs @ direction) * direction`).

## Interpretation tips

- **Sweep coefficient and layer.** A working steering direction has a smooth band of `(layer, coef)` where the behavior shifts and fluency is preserved. Outside this band the model degrades.
- **Look at fluency, not just argmax.** A direction can flip the top token but produce gibberish for the next 10 tokens. Always inspect a generation, not just the next-token logits.
- **Last position vs all positions** changes the effect. Last-position steering is more surgical; all-position is stronger but more disruptive.
- **Norm-normalize the direction.** `direction / direction.norm()` makes coefficient comparisons across directions meaningful.
- **Compare to a random direction baseline** at the same norm. A real concept direction should beat a random vector of the same magnitude.
- **Position of the steering layer matters.** Early-layer steering is mostly "hey, prepend this concept". Mid/late-layer steering is closer to "produce text with this concept". Late-late steering tends to corrupt grammar.

## Gotchas

- `+= direction * coef` mutates the residual in place. Save a `clone()` before the addition if you need the pre-steer state.
- Device placement: `direction.to(model.device)` if you computed it elsewhere. See `docs/usage/access-and-modify.md`.
- For models like Llama, the residual at layer L lives at `model.model.layers[L].output[0]`, not `model.transformer.h[L].output[0]`. Use `print(model)`.
- Steering one layer's output is *cumulative* downstream - every later layer reads the modified residual. If you want to test "does layer L need this concept added?", use activation patching instead.

## Related

- [activation-patching](activation-patching.md)
- [ablation](ablation.md)
- [multi-prompt-comparison](multi-prompt-comparison.md)
- `docs/usage/session.md` - Bundling the direction-computation step and the steering step into one session (one remote round-trip).
- Turner et al. (2023), "Activation Addition".
- Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single Direction".

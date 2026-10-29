---
title: Steering with Added Vectors
one_liner: Add a precomputed direction to the residual stream at a specific layer to push the model's behavior in a target direction.
tags: [pattern, interpretability, steering, residual-stream]
related: [docs/usage/access-and-modify.md, docs/usage/generate.md, docs/usage/iteration.md, docs/patterns/ablation.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Steering with Added Vectors

## What this is for

Activation steering (activation addition, ActAdd) changes behavior by adding a fixed
vector to a layer's residual stream during the forward pass — usually the difference
of mean activations between contrasting prompt sets, or a probe direction. Adding it
pushes toward the concept; subtracting suppresses it. Use it to test whether a
behavior is linearly steerable from a layer, or to check that a probe direction is
causal. Refusal-direction work (Arditi et al., 2024) is the same pattern with
`mean(harmful) - mean(harmless)` subtracted or projected out. In nnsight it is an
in-place `+=` to `block.output` inside a trace.

## Computing the steering direction

Take the difference of mean residuals between a positive and a negative prompt set
at the chosen layer:

```python
import torch
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

positive = ["I love this so much", "This is wonderful", "I am very happy"]
negative = ["I hate this so much", "This is terrible", "I am very sad"]
LAYER = 6

with model.trace() as tracer:
    pos_acts = nnsight.save([])
    neg_acts = nnsight.save([])
    for p in positive:
        with tracer.invoke(p):
            pos_acts.append(model.transformer.h[LAYER].output[:, -1, :])
    for p in negative:
        with tracer.invoke(p):
            neg_acts.append(model.transformer.h[LAYER].output[:, -1, :])

direction = torch.cat(list(pos_acts)).mean(0) - torch.cat(list(neg_acts)).mean(0)
direction = (direction / direction.norm()).detach()     # [768]
```

`.detach()` matters: saved activations carry `requires_grad=True`, so an attached
vector drags its autograd graph into every run you add it to.

## Canonical pattern

Add the direction at one layer and compare the next-token prediction:

```python
coef = 8.0
prompt = "I went to the bakery and"

with model.trace() as tracer:
    with tracer.invoke(prompt):
        baseline = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(prompt):
        model.transformer.h[LAYER].output[:, -1, :] += direction * coef
        steered = model.lm_head.output[:, -1, :].save()

print("baseline argmax:", repr(model.tokenizer.decode(baseline.argmax(-1)[0])))
print("steered  argmax:", repr(model.tokenizer.decode(steered.argmax(-1)[0])))
```

```
baseline argmax: ' bought'
steered  argmax: ' I'
```

`block.output` is a plain tensor, so `+=` writes the residual in place. Neither invoke
reads a value from the other, so no barrier is needed.

## Choosing the coefficient

No absolute coefficient transfers, because the quantity you are adding to grows
with depth. On the prompt below, the last-position residual norm runs 57 at block
0, 92 at block 6 and 465 at block 11. A coefficient that is a light nudge late is
a demolition early.

Measure the norm at the layer you are steering and read the coefficient as a
fraction of it:

```python
with model.trace(prompt):
    scale = model.transformer.h[LAYER].output[0, -1].norm().detach().save()

print(f"residual norm {float(scale):.0f}; coef {coef} is {coef / float(scale):.2f}x it")
# residual norm 92; coef 8.0 is 0.09x it
```

The canonical example above is a light nudge by that measure. Sweep the fraction,
not the raw number. GPT-2 holds together well past the point most tutorials warn
about:

```
coef   10  (0.11x): 'I went to the bakery and I was told that the bread was good. I was very happy with it.'
coef   24  (0.26x): 'I went to the bakery and I was surprised to find that the bread was very good. I also found that'
coef   47  (0.51x): 'I went to the bakery and I found a little bit of the butter and the salt. I also added the'
coef   93  (1.01x): 'I went to the bakery and I was very happy with the store. I have the same size and I have'
coef  186  (2.02x): 'I went to the bakery and I was the same you and I you and you and you and you and you'
```

Fluency holds to 93 and breaks at 186 — a little past one times the residual norm.
At block 10, where the norm is 244, the same fraction is `coef = 122`. Sweep the
layer alongside the fraction; the two interact.

## Steering during generation

A bare edit in a `generate` body fires **once, on the prefill** — the forward that
encodes the prompt — and not on the decode steps. That still changes the output, so
it is easy to mistake for working. To keep the steering on while the model writes,
put the edit in a **bounded** `tracer.iter[:N]`:

```python
with model.generate(prompt, max_new_tokens=20, min_new_tokens=20, do_sample=False) as tracer:
    for _ in tracer.iter[:20]:                       # every step, prompt included
        model.transformer.h[LAYER].output[:, -1, :] += direction * coef
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
```

Bounded, not `tracer.all()`: an open-ended loop unwinds every line after it, so
`tracer.result.save()` would never run. `min_new_tokens` pins the run to the bound —
steering changes when the model emits end-of-text, and a loop the run stops short
of is cut short with a warning and `ids` is never bound. See
[iteration](../usage/iteration.md).

With the direction and `coef = 8.0` above, the two diverge on the third generated
token:

```
baseline     :  bought a bag of cookies. I was so excited. I
prefill only :  I was like, 'Oh my God, I'm so
every step   :  I was told that the bread was good. I was told
```

The prefill-only run *does* differ from the baseline, which is why a missing
per-step application reads as "the effect is just modest" rather than as a bug.

To steer only part of the generation, slice the steps. Step 0 is the prefill, so
this applies the vector on decode steps 5–10; a write at layer L on step k reaches
the layers above L at once, so it can change the token step k emits, and carries to
later steps through their KV cache. Layers ≤ L stay bit-identical to the unsteered
run until a generated token actually differs (see
[iteration](../usage/iteration.md#steps-vs-tokens)):

```python
with model.generate(prompt, max_new_tokens=20, min_new_tokens=20, do_sample=False) as tracer:
    for step in tracer.iter[5:11]:
        model.transformer.h[LAYER].output[:, -1, :] += direction * coef
    ids = tracer.result.save()
```

With the KV cache, each decode step's activation covers only the token just
produced, so `[:, -1, :]` is the last prompt token on step 0 and the new token after;
an absolute index like `[:, 5, :]` is out of range from step 1 on.

On `VLLM`, indexing, the `(hidden, residual)` layer output and how to scale the vector
differ; see [What your block sees on vLLM](../models/vllm.md#what-your-block-sees-on-vllm).

## Interpretation tips

- **Look at fluency, not just argmax.** A direction can flip the top token and
  produce gibberish downstream — always inspect a generation.
- **Compare to a random direction** of the same norm — a real concept direction
  should beat it.
- **Last position vs all positions** (`output[:] += ...`): last-position is more
  surgical; all-position is stronger but more disruptive.

## Gotchas

- Device placement: `direction.to(model.device)` if computed elsewhere.
- For Llama-family, the residual is at `model.model.layers[L].output`, not
  `model.transformer.h[L].output`. Use `print(model)`.
- Steering one layer is *cumulative* downstream — every later layer reads the
  modified residual. To test "does layer L *need* this concept added?", use
  activation patching instead.

## Related

- [activation-patching](activation-patching.md)
- [ablation](ablation.md)
- [invoke-and-batching](../usage/invoke-and-batching.md#comparing-prompts-one-trace-not-a-loop-of-traces)
- [generate](../usage/generate.md) — generate returns token ids via `tracer.result`.
- [iteration](../usage/iteration.md) — step-conditional steering during generation.
- Turner et al. (2023), "Activation Addition".
- Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single
  Direction".

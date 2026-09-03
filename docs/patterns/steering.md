---
title: Steering with Added Vectors
one_liner: Add a precomputed direction to the residual stream at a specific layer to push the model's behavior in a target direction.
tags: [pattern, interpretability, steering, residual-stream]
related: [docs/usage/access-and-modify.md, docs/usage/generate.md, docs/usage/iter-all-next.md, docs/patterns/ablation.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Steering with Added Vectors

## What this is for

Activation steering (a.k.a. "activation addition", ActAdd) changes behavior by
adding a fixed vector to a layer's residual stream during the forward pass. The
vector is usually a "concept direction" — the difference of mean activations
between contrasting prompt sets — or a learned probe direction. Adding it pushes
the residual toward that concept; subtracting suppresses it.

The same machinery drives refusal-direction work, sentiment / topic steering, and
"control vectors". The interpretability claim: if a single low-rank addition to one
layer reliably changes behavior, that behavior has a linearly-decodable
representation at that layer.

In nnsight this is an in-place `+=` to `block.output` inside a trace, plus a
separate computation of the direction.

## When to use

- Testing whether a behavior is linearly steerable from a given layer.
- Comparing steering effectiveness across layers / coefficients.
- Building refusal-direction / sycophancy / persona interventions.
- Re-injecting a probe's learned direction to measure it.

## Computing the steering direction

Take the difference of mean residuals between a positive and negative prompt set at
the chosen layer. Bundle both passes in one `session()` (one engine call if remote;
cross-trace values flow without a per-element `.save()`):

```python
import torch
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

positive = ["I love this so much", "This is wonderful", "I am very happy"]
negative = ["I hate this so much", "This is terrible", "I am very sad"]
LAYER = 6

with model.session():
    with model.trace() as tracer:
        pos_acts = nnsight.save([])
        neg_acts = nnsight.save([])
        for p in positive:
            with tracer.invoke(p):
                pos_acts.append(model.transformer.h[LAYER].output[:, -1, :])
        for p in negative:
            with tracer.invoke(p):
                neg_acts.append(model.transformer.h[LAYER].output[:, -1, :])

pos = torch.cat([a for a in pos_acts], dim=0).mean(0)   # [hidden]
neg = torch.cat([a for a in neg_acts], dim=0).mean(0)
direction = (pos - neg)
direction = direction / direction.norm()                # [768]
```

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

`block.output` is a plain tensor, so `+=` writes the residual in place. Two invokes
write disjoint variables and only the second modifies the steered module, so **no
barrier is needed**.

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

## Variations

### Steer all positions vs only the last

```python
# All positions: nudges every token's residual.
model.transformer.h[LAYER].output[:] += direction * coef

# Only the last position: cleaner targeting on next-token prediction.
model.transformer.h[LAYER].output[:, -1, :] += direction * coef

# A specific span (positions 5-10): targeted intervention.
model.transformer.h[LAYER].output[:, 5:10, :] += direction * coef
```

### Multi-layer steering

Adding the same direction at several consecutive layers often produces stronger,
cleaner effects than a single big push:

```python
with model.trace(prompt):
    for L in [4, 5, 6, 7]:
        model.transformer.h[L].output[:, -1, :] += direction * (coef / 4)
    out = model.lm_head.output.save()

# out[:, -1, :].argmax -> ' I'
```

### Steering during generation

`model.generate(...)` returns generated **token ids** (read `tracer.result`).

A bare edit in the trace body fires **once, on the prefill** — the forward that
encodes the prompt — and not on the decode steps that follow. That still changes
the output, because it changes how the prompt was read, so it is easy to mistake
for working. To keep the steering on while the model writes, put the edit in a
**bounded** `tracer.iter[:N]`:

```python
with model.generate(prompt, max_new_tokens=20, min_new_tokens=20, do_sample=False) as tracer:
    for _ in tracer.iter[:20]:                       # every step, prompt included
        model.transformer.h[LAYER].output[:, -1, :] += direction * coef
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
```

Bounded, not `tracer.all()`: an open-ended loop unwinds every line after it, so
`tracer.result.save()` would never run (see
[iter-all-next.md](../usage/iter-all-next.md)).

`min_new_tokens` matches the bound to what the run will actually do. Steering
changes when the model emits an end-of-text token, and a bounded loop the run
stops short of raises `OutOfOrderError` naming the iteration it asked for. Pinning
the step count also keeps the rows of a sweep the same length.

The two are easy to tell apart once you look. They agree while the prefill's
effect is still carrying, then diverge once only one of them is still adding
anything: here on the third generated token, `' told'` against `' like'`. With the
direction and `coef = 8.0` above, continuing `"I went to the bakery and"`:

```
baseline     :  bought a bag of cookies. I was so excited. I
prefill only :  I was like, 'Oh my God, I'm so
every step   :  I was told that the bread was good. I was told
```

Note that the prefill-only run *does* differ from the baseline — the prompt was
encoded differently — which is why a missing per-step application reads as "the
steering is working, the effect is just modest" rather than as a bug.

#### On `VLLM`

Activations have no batch axis and a decoder layer returns `(hidden, residual)`:
`model.model.layers[L].output[0][-1] += direction * coef`, with `direction` moved onto the
served tensor's device inside the block. Scale against the *median* per-token residual norm — the
first position is often an attention sink with a norm 100× the rest, and on 8B-class models ~1×
that median already saturates. See [What your block sees on vLLM](../models/vllm.md#what-your-block-sees-on-vllm).

#### `[:, -1, :]` means something different after the prefill

With `use_cache=True` (the default), the prefill's activation covers the whole
prompt, but each decode step's covers **only the token just produced** — sequence
length 1. So `[:, -1, :]` is the last prompt token on step 0 and the new token on
every step after, which is what you want here.

An **absolute** index is not portable that way: `[:, 5, :]` addresses prompt
position 5 during the prefill and is out of range from step 1 on. If you are
steering a fixed prompt position rather than the running token, apply it outside
the loop — or index from the end.

For step-conditional steering (e.g. only the first 5 tokens), slice the range:
`tracer.iter[:5]`. See [iter-all-next.md](../usage/iter-all-next.md).

### Refusal direction

The "refusal direction" line of work computes
`mean(harmful) - mean(harmless)` at a middle layer and *subtracts* it (or projects
it out) to suppress refusal. Same pattern with `coef < 0` or projection
(`hs - (hs @ direction) * direction`).

## Interpretation tips

- **Sweep coefficient and layer.** A working direction has a band of `(layer, coef)`
  where behavior shifts and fluency holds. Outside it the model degrades.
- **Look at fluency, not just argmax.** A direction can flip the top token but
  produce gibberish downstream — always inspect a generation.
- **Last position vs all positions.** Last-position is more surgical; all-position
  is stronger but more disruptive.
- **Norm-normalize the direction** so coefficients are comparable across directions.
- **Compare to a random direction** at the same norm — a real concept direction
  should beat it.
- **Layer position matters.** Early-layer steering "prepends a concept"; mid/late
  steering shapes the text; late-late steering tends to corrupt grammar.

## Gotchas

- `+= direction * coef` mutates the residual in place. `.clone()` before if you need
  the pre-steer state.
- Device placement: `direction.to(model.device)` if computed elsewhere.
- For Llama-family, the residual is at `model.model.layers[L].output`, not
  `model.transformer.h[L].output`. Use `print(model)`.
- Steering one layer is *cumulative* downstream — every later layer reads the
  modified residual. To test "does layer L *need* this concept added?", use
  activation patching instead.

## Related

- [activation-patching](activation-patching.md)
- [ablation](ablation.md)
- [multi-prompt-comparison](multi-prompt-comparison.md)
- `docs/usage/generate.md` — generate returns token ids via `tracer.result`.
- `docs/usage/iter-all-next.md` — step-conditional steering during generation.
- Turner et al. (2023), "Activation Addition".
- Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single
  Direction".

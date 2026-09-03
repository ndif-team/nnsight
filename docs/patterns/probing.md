---
title: Probing
one_liner: Train a linear classifier on activations to test what is decodable from a model's internal state, where in depth it appears, and whether the model uses it.
tags: [pattern, interpretability, probing, residual-stream, classifier]
related: [docs/patterns/logit-lens.md, docs/patterns/steering.md, docs/usage/invoke-and-batching.md, docs/usage/cache.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Probing

## What this is for

A probe asks whether a property is **linearly decodable** from a model's internal
state: collect activations for a labelled dataset, fit a classifier, report
accuracy per layer. The curve says where in depth the property becomes readable.

A probe measures what *you* can decode, not what the model computes or uses. The
controls below are what separate those, and they are the part of the recipe most
often skipped.

## When to use

- Testing whether a property (truth, sentiment, entity type, an eventual answer)
  is present in the residual stream, and from which layer.
- Extracting a direction to steer along — see [steering](steering.md).
- Comparing two prompts sets, two positions, or two sublayers on the same property.
- Screening before a causal method: a probe is cheap, patching is not.

## Canonical pattern

The whole dataset and every layer come from one batched forward pass.

```python
import torch
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
n_layers = len(model.transformer.h)

positive_words = ["wonderful", "fantastic", "delightful", "excellent", "brilliant",
                  "joyful", "superb", "enjoyable", "lovely", "amazing",
                  "great", "charming", "pleasant", "terrific", "marvelous", "splendid"]
negative_words = ["terrible", "awful", "dreadful", "disgusting", "horrible",
                  "miserable", "dismal", "boring", "dull", "abysmal",
                  "bad", "unpleasant", "annoying", "atrocious", "lousy", "painful"]
templates = ["The movie was {}.", "I found the book {}.",
             "That meal was {}.", "Their performance was {}."]

texts = ([t.format(w) for w in positive_words for t in templates]
         + [t.format(w) for w in negative_words for t in templates])
labels = torch.tensor([1.0] * 64 + [0.0] * 64)

with torch.no_grad():                              # see Gotchas — not optional
    with model.trace(texts):
        activations = nnsight.save([
            block.output[:, -1, :].detach().cpu().float()
            for block in model.transformer.h
        ])

assert len(activations) == n_layers
assert activations[0].shape == (len(texts), model.config.n_embd)
```

Then one logistic regression per layer, with weight decay and a train/test split:

```python
generator = torch.Generator().manual_seed(0)
order = torch.randperm(len(texts), generator=generator)
split = int(0.7 * len(texts))
train_idx, test_idx = order[:split], order[split:]

def train_probe(features, y, tr, te, steps=300, l2=0.02):
    mean, std = features[tr].mean(0), features[tr].std(0) + 1e-6
    x_train, x_test = (features[tr] - mean) / std, (features[te] - mean) / std

    weight = torch.zeros(features.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.02)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            x_train @ weight + bias, y[tr]
        ) + l2 * weight.pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = (((x_test @ weight + bias) > 0).float() == y[te]).float().mean()
    return float(accuracy), weight.detach()

for layer in range(0, n_layers, 2):
    accuracy, _ = train_probe(activations[layer], labels, train_idx, test_idx)
    print(f"layer {layer:2d}  test accuracy {accuracy:.2f}")
```

```
layer  0  test accuracy 1.00
layer  2  test accuracy 1.00
layer  4  test accuracy 1.00
layer  6  test accuracy 1.00
layer  8  test accuracy 1.00
layer 10  test accuracy 1.00
```

**Perfect accuracy at layer 0 is a red flag, not a result.** Layer 0 has barely
processed anything, so a probe that succeeds there is reading token identity:
"wonderful" and "terrible" are different tokens with different embeddings. Read the
shape of the curve rather than its height. High from layer 0 means a surface
feature; a rise through the middle means something the model computes; high only
at the end may be the prediction itself.

## Controls

Each of these kills a different false positive, and each costs one more fit.

**Shuffled labels** — destroys the real relationship. Anything above chance is
memorization capacity, which means the probe is too expressive for the dataset:

```python
shuffled = labels[torch.randperm(len(labels), generator=generator)]
control, _ = train_probe(activations[6], shuffled, train_idx, test_idx)
print(f"shuffled labels at layer 6: {control:.2f}   (chance = 0.50)")

assert control < 0.6            # measured: 0.46
```

**Held-out template** — train on three templates, test on the fourth. If accuracy
collapses, the probe learned the template rather than the property:

```python
held_out = torch.tensor([i for i in range(len(texts)) if i % len(templates) == 3])
kept = torch.tensor([i for i in range(len(texts)) if i % len(templates) != 3])
accuracy, _ = train_probe(activations[6], labels, kept, held_out)
print(f"held-out template at layer 6: {accuracy:.2f}")      # 1.00
```

**A random direction of the same norm** is the floor for any claim that a
direction *encodes* something. It appears under [Causal validation](#causal-validation)
below, where it does the most work.

On this dataset the controls pass and the honest summary is still deflationary:
the probe has learned the adjective vocabulary, which generalizes across templates
and is readable from the embeddings up. To probe something the model has to
*compute*, build a dataset where no surface cue predicts the label — entailment
between two sentences, whether a stated fact is true, the answer to a question
asked later.

## Variations

### Difference-in-means (mass-mean) probe

No fitting: the direction is the difference between the class means. It is more
robust on small datasets and tends to be more causally effective than a trained
probe, which is free to use directions that discriminate without being the ones
the model acts on.

```python
LAYER = 6

def mass_mean_direction(features, index):
    positive = features[index][labels[index] == 1].mean(0)
    negative = features[index][labels[index] == 0].mean(0)
    direction = positive - negative
    return direction / direction.norm()

direction = mass_mean_direction(activations[LAYER], train_idx)
projections = activations[LAYER][test_idx] @ direction
threshold = (activations[LAYER][train_idx] @ direction).mean()
accuracy = ((projections > threshold).float() == labels[test_idx]).float().mean()

print(f"difference-in-means at layer {LAYER}: {float(accuracy):.2f}")     # 1.00
```

### Other sites

`block.output` at the last position is the default. The same collection works on
`block.mlp.output` (what was computed), `block.attn.output[0]` (what was moved),
and any token position — `[:, -1, :]` is a choice, and for a property mentioned
mid-prompt it is usually the wrong one.

### Larger datasets

For a dataset too large for one batch, chunk the texts and concatenate the
results — still one forward pass per chunk, never one pass per layer.
`tracer.cache(modules=...)` is the alternative when you want inputs and outputs of
many modules at once; see `docs/usage/cache.md`.

## Causal validation

A probe is correlational. To claim the model *uses* the direction, add it to the
residual stream and check that the behavior moves — and check that a random
direction of the same norm does not.

```python
scale = 0.5
probe_direction = direction.to(model.device)

test_prompt = "The movie was"
good = model.tokenizer.encode(" great")[0]
bad = model.tokenizer.encode(" bad")[0]

def logit_gap(vector):
    """logit(' great') - logit(' bad') after adding `vector` at LAYER, or nothing."""
    with model.trace(test_prompt):
        if vector is not None:
            norm = model.transformer.h[LAYER].output[0, -1].norm()
            model.transformer.h[LAYER].output[:, -1, :] += scale * norm * vector
        logits = model.output.logits[0, -1]
        gap = (logits[good] - logits[bad]).detach().save()
    return float(gap)

baseline = logit_gap(None)
steered = logit_gap(probe_direction)
negated = logit_gap(-probe_direction)

random_generator = torch.Generator().manual_seed(0)
random_gaps = []
for _ in range(8):
    noise = torch.randn(probe_direction.shape, generator=random_generator)
    random_gaps.append(logit_gap((noise / noise.norm()).to(model.device)))

print(f"baseline {baseline:+.3f}   probe {steered:+.3f}   negated {negated:+.3f}")
print(f"random directions: mean {sum(random_gaps) / len(random_gaps):+.3f}"
      f"   max {max(random_gaps):+.3f}")

assert steered > baseline and negated < baseline
assert steered > max(random_gaps)
```

```
baseline +1.129   probe +5.315   negated -2.950
random directions: mean +1.020   max +1.846
```

The probe direction moves the gap by `+4.19` and its negation by `-4.08`, while
eight random directions of the same norm land between `-0.02` and `+1.85`,
straddling the baseline. Without that last line the first two numbers would be consistent with
"any perturbation of this size moves the logits".

A direction that steers nothing is also a result: a decodable feature the model
does not read. Report it rather than tuning the scale until something moves — for
which band to sweep, see [steering](steering.md).

## Interpretation tips

- **Report the curve, not the maximum.** "Accuracy 0.98" without the layer axis
  and the layer-0 value is not interpretable.
- **Balance the classes and say how many examples.** Below a few hundred, treat
  any probe result as provisional.
- **Keep weight decay on and report it.** With `hidden` ≫ examples an
  unregularized probe fits anything.
- **Accuracy is not comparable across models** with different hidden sizes unless
  you hold the probe's capacity fixed.

## Gotchas

- **Wrap collection in `torch.no_grad()`.** A trace runs with autograd on, so a
  saved activation comes back with a live `grad_fn` pinning the whole forward
  graph. This 128-example, 12-layer capture peaks at **279 MiB** of activation
  memory above the weights under `no_grad` and **1293 MiB** without it, for
  bit-identical values. `.detach()` on the saved tensor does not help: the graph
  is already built by then.
- **A block's `.output` is a plain tensor**, so `block.output[0]` selects batch
  row 0 — a silent 1-of-N dataset, not a layer. See
  [logit-lens](logit-lens.md#check-the-wiring).
- **Collect in forward order.** Reading layer 9 and then layer 7 in one trace
  raises `OutOfOrderError`; the list comprehension above is in order by
  construction.
- **Fit on training rows only**, including the standardization statistics. A
  probe standardized with test-set means has already seen the answer.

## Related

- [steering](steering.md) — turning a probe direction into an intervention.
- [logit-lens](logit-lens.md) — the other read-only depth measurement.
- [activation-patching](activation-patching.md) — component-level causality.
- [sae-and-auxiliary-modules](sae-and-auxiliary-modules.md) — unsupervised
  features instead of supervised probes.
- `docs/usage/cache.md` — collecting many modules at once.

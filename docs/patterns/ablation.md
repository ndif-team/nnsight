---
title: Ablation
one_liner: Zero, mean, or noise the output of a component to measure its functional contribution.
tags: [pattern, interpretability, ablation, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/invoke-and-batching.md, docs/patterns/activation-patching.md, docs/patterns/per-head-attention.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Ablation

## What this is for

Ablation removes (or replaces with a baseline) the output of a component and
measures how the prediction degrades. It is the "lesion study" of mechanistic
interpretability: if zeroing a component destroys a behavior, that component is
doing something the task needs.

Common variants:

- **Zero ablation**: replace the output with zeros. Simple, but pushes the residual
  off-distribution.
- **Mean ablation**: replace with the average activation over a reference set. Stays
  closer to typical model state.
- **Noise / resampling ablation**: replace with a sample from another input.

In nnsight, ablation is an in-place write inside a trace. Pair an ablated invoke
with a baseline invoke to compare.

## When to use

- Identifying which components a behavior depends on.
- Validating a circuit hypothesis.
- Measuring component importance for layer / head pruning.
- Sanity-checking probes: if you can read X off layer L but ablating L doesn't
  change the output, X may be epiphenomenal.

## Canonical pattern

Zero-ablate one MLP block, compare to baseline:

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
LAYER = 9
paris = model.tokenizer.encode(" Paris")[0]

with model.trace() as tracer:
    with tracer.invoke(prompt):                       # baseline
        baseline_logits = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(prompt):                       # ablated
        model.transformer.h[LAYER].mlp.output[:] = 0
        ablated_logits = model.lm_head.output[:, -1, :].save()

print(f"baseline P(Paris) = {baseline_logits.softmax(-1)[0, paris]:.3f}")
print(f"ablated  P(Paris) = {ablated_logits.softmax(-1)[0, paris]:.3f}")
```

```
baseline P(Paris) = 0.070
ablated  P(Paris) = 0.053
```

`mlp.output` is a plain tensor, so `[:] = 0` writes in place. The two invokes write
to disjoint variables and never read a shared module across invokes, so **no barrier
is needed**.

## Variations

### Zero a specific position only

```python
with model.trace(prompt):
    model.transformer.h[LAYER].mlp.output[:, -1, :] = 0
    logits = model.lm_head.output[:, -1, :].save()
```

### Zero a single hidden dimension / feature

```python
with model.trace(prompt):
    feature_idx = 1234
    model.transformer.h[LAYER].mlp.output[:, :, feature_idx] = 0
    logits = model.lm_head.output[:, -1, :].save()
```

### Zero one attention head

`.attn.output` is a tuple `(attn_out, weights)`. Index `[0]` for the value-weighted
output `[batch, seq, hidden]`, reshape to heads, zero one, and assign the whole tuple
back:

```python
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads

with model.trace(prompt):
    out = model.transformer.h[LAYER].attn.output
    attn_out = out[0]                               # [B, S, hidden]
    B, S, _ = attn_out.shape
    reshaped = attn_out.view(B, S, n_heads, head_dim).clone()
    reshaped[:, :, 4, :] = 0                         # zero head 4
    new_attn = reshaped.view(B, S, n_heads * head_dim)
    model.transformer.h[LAYER].attn.output = (new_attn,) + tuple(out[1:])
    logits = model.lm_head.output[:, -1, :].save()

# P(Paris) = 0.066  (down slightly from 0.070 baseline)
```

For ergonomic per-head access, see `docs/patterns/per-head-attention.md`.

### Mean ablation

Collect the mean activation over a reference set, then write it in:

```python
import torch
import nnsight

ref_prompts = ["The capital of France is", "Paris is in", "Berlin lies in", "London hosts the"]

# Pass 1: collect mean MLP activations at LAYER, last position.
with model.trace() as tracer:
    acts = nnsight.save([])
    for p in ref_prompts:
        with tracer.invoke(p):
            acts.append(model.transformer.h[LAYER].mlp.output[:, -1, :])

mean_act = torch.stack([a for a in acts], dim=0).mean(dim=0)   # [1, hidden]

# Pass 2: ablate with the mean.
with model.trace(prompt):
    model.transformer.h[LAYER].mlp.output[:, -1, :] = mean_act
    logits = model.lm_head.output[:, -1, :].save()

# mean-ablated P(Paris) = 0.078
```

Create the accumulator inside the trace with `nnsight.save([])` and append raw
values so the collected activations survive the trace.

### Noise / resampling ablation

Replace with the activation from an unrelated prompt. Both invokes touch the same
module and one passes a value to the other, so a `tracer.barrier(n)` is required —
see `docs/usage/barrier.md`.

```python
clean     = "The Eiffel Tower is in the city of"
unrelated = "I went to the store and bought some"

with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke(unrelated):
        noise = model.transformer.h[LAYER].mlp.output[:, -1, :].save()
        barrier()
    with tracer.invoke(clean):
        barrier()
        mlp = model.transformer.h[LAYER].mlp.output
        mlp[:, -1, :] = noise
        model.transformer.h[LAYER].mlp.output = mlp
        logits = model.lm_head.output[:, -1, :].save()

# resample-ablated P(Paris) = 0.037
```

### Empty invoke for batch-wide ablation

To ablate uniformly across a multi-invoke batch, add an empty `tracer.invoke()` — it
covers the full batch:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):
        pass
    with tracer.invoke("The Colosseum is in the city of"):
        pass
    with tracer.invoke():                       # empty: full batch
        model.transformer.h[LAYER].mlp.output[:] = 0
        logits = model.lm_head.output[:, -1, :].save()   # shape [2, 50257]
```

See `docs/usage/invoke-and-batching.md`.

## Interpretation tips

- **Compare logits, log-probs, or task accuracy** — not raw activation norms. The
  point is functional effect.
- **Zero ablation pushes out of distribution.** A large drop might just mean
  "anything missing here breaks the model". Mean / resample ablation is more
  conservative.
- **One-component ablation underestimates redundancy.** If two heads back each
  other up, zeroing either alone barely moves the answer. Ablate sets jointly.
- **Different ablations answer different questions.** Zero = "is there *any* signal
  here?" Mean = "is the *deviation from average* important?" Resample = "is the
  *task-specific* content important?"

## Gotchas

- `[:] = 0` is in-place. `.clone().save()` first if you need the pre-ablation value.
- Submodule output shapes vary: `.mlp.output` is a tensor; `.attn.output` is a
  tuple `(attn_out, weights)` — index `[0]` and rebuild `(new,) + tuple(out[1:])`
  when replacing. GPT-2 **block** outputs are plain tensors in current
  `transformers`. Check with `isinstance(module.output, tuple)`. See
  `docs/usage/access-and-modify.md`.
- Accumulating activations into a list across invokes: create the list inside the
  trace as `xs = nnsight.save([])` and append raw values.
- Module names differ across architectures — `print(model)` to inspect.
- If two invokes read *and* write the same module's output, use `tracer.barrier(n)`.
  See `docs/usage/barrier.md`.

## Related

- [activation-patching](activation-patching.md) — the opposite: paste in a clean
  activation instead of zeroing.
- [attribution-patching](attribution-patching.md)
- [per-head-attention](per-head-attention.md)
- [multi-prompt-comparison](multi-prompt-comparison.md)
- `docs/usage/access-and-modify.md`
- `docs/usage/barrier.md`

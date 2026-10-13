---
title: Ablation
one_liner: Zero, mean, or noise the output of a component to measure its functional contribution.
tags: [pattern, interpretability, ablation, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/invoke-and-batching.md, docs/patterns/activation-patching.md, docs/patterns/per-head-attention.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Ablation

## What this is for

Remove a component's output, or replace it with a baseline, and measure how the
prediction degrades. If removing it breaks a behavior, the behavior needs it — use
this to test a circuit hypothesis or to check that a probed feature is actually used.

- **Zero**: replace with zeros. Simple, but pushes the residual off-distribution.
- **Mean**: replace with the average activation over a reference set.
- **Resample**: replace with the activation from another input.

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

`mlp.output` is a plain tensor, so `[:] = 0` writes in place. Neither invoke reads a
value from the other, so no barrier is needed.

## Variations

### Zero one position or one neuron

Index the write. `mlp.output` is `[B, S, hidden]`; individual MLP neurons live on
`mlp.act.output` (`[B, S, 4*hidden]` on GPT-2). Either line works alone; together
they must stay in forward order (`act` runs inside the MLP, before its output):

```python
with model.trace(prompt):
    model.transformer.h[LAYER].mlp.act.output[:, -1, 1234] = 0   # one neuron
    model.transformer.h[LAYER].mlp.output[:, -1, :] = 0          # last position only
    logits = model.lm_head.output[:, -1, :].save()
```

### Zero one attention head

Zero the head's slice of the output projection's *input*, where the hidden dimension
still decomposes per head. Do not reshape `attn.output[0]`: that is post-projection,
and its column slices are not heads — zeroing "layer 0 head 10" there leaves
P(Paris) at `0.067`, barely moved, where the correct slice below drops it to `0.0059`
(see [per-head-attention](per-head-attention.md)).

```python
HEAD = 4
# read it from the module — equals n_embd // n_head on GPT-2, but not on GQA/Gemma-style models
head_dim = model.transformer.h[0].attn._module.head_dim

with model.trace(prompt):
    lo, hi = HEAD * head_dim, (HEAD + 1) * head_dim
    model.transformer.h[LAYER].attn.c_proj.input[:, :, lo:hi] = 0
    logits = model.lm_head.output[:, -1, :].save()

# P(Paris) = 0.065  (baseline 0.070)
```

### Sweep every head

One invoke per (layer, head) plus a baseline, all in one trace — one batched forward
pass instead of 144 (0.12 s against 3.12 s for a loop of traces):

```python
import nnsight

with model.trace() as tracer:
    probs = nnsight.save({})
    with tracer.invoke(prompt):
        probs["baseline"] = model.lm_head.output[0, -1].softmax(-1)[paris].item()
    for layer in range(len(model.transformer.h)):
        for head in range(model.config.n_head):
            with tracer.invoke(prompt):
                model.transformer.h[layer].attn.c_proj.input[:, :, head * head_dim:(head + 1) * head_dim] = 0
                probs[layer, head] = model.lm_head.output[0, -1].softmax(-1)[paris].item()

# baseline 0.070; largest effects: (0, 10) -> 0.0059, (0, 0) -> 0.1105 (ablating it *raises* P)
```

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

mean_act = torch.stack(list(acts), dim=0).mean(dim=0)   # [1, hidden]

# Pass 2: ablate with the mean.
with model.trace(prompt):
    model.transformer.h[LAYER].mlp.output[:, -1, :] = mean_act
    logits = model.lm_head.output[:, -1, :].save()

# mean-ablated P(Paris) = 0.078
```

### Noise / resampling ablation

Replace with the activation from an unrelated prompt. Both invokes touch the same
module and one passes a value to the other, so a `tracer.barrier(n)` is required —
see [barrier](../usage/barrier.md).

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
        model.transformer.h[LAYER].mlp.output[:, -1, :] = noise
        logits = model.lm_head.output[:, -1, :].save()

# resample-ablated P(Paris) = 0.037
```

### Ablating an MoE expert

On vLLM, mask the expert's router logit: `mlp.gate.output[0][:, e] = -inf` (see
[vllm-parallelism.md](../models/vllm-parallelism.md)).

On **`TransformersModel`** the same line is a silent no-op. A transformers MoE block
calls its router as `_, top_k_weights, top_k_index = self.gate(hidden_states)`, and
element `[0]`, the logits, is discarded, so masking it changes nothing: setting all 64
logits to `-inf` gives `max|delta| = 0.0`. Edit the *selection* there instead, by
writing the routing weights or expert indices in `mlp.gate.output[1]` / `[2]`.

Two further traps on that side. With `norm_topk_prob=False`, masking a weight rescales
every surviving expert, so a never-selected expert can show a larger effect than a real
below-median one; ablate by selection, not by weight. And the router's `.input` /
`.output` have no batch axis (`(B*T, E)`), so per-token statistics over a padded batch
quietly include the pad rows.

## Interpretation tips

- **Compare logits, log-probs, or task accuracy**, not activation norms.
- **A large zero-ablation drop may only mean "anything missing here breaks the
  model".** Mean ablation asks whether the *deviation from average* matters;
  resample asks whether the *task-specific* content does.
- **One-component ablation underestimates redundancy.** If two heads back each
  other up, zeroing either alone barely moves the answer. Ablate sets jointly.

## Gotchas

- Module names differ across architectures — `print(model)` to inspect.

## Related

- [activation-patching](activation-patching.md) — the opposite: paste in a clean
  activation instead of zeroing.
- [attribution-patching](attribution-patching.md)
- [per-head-attention](per-head-attention.md)
- [invoke-and-batching](../usage/invoke-and-batching.md#comparing-prompts-one-trace-not-a-loop-of-traces)
- `docs/usage/access-and-modify.md`
- `docs/usage/barrier.md`

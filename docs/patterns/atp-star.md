---
title: AtP* Building Blocks
one_liner: Build attribution patching with local Q/K corrections, GradDrop cancellation checks, and exact verification.
tags: [pattern, interpretability, gradients, attribution, patching, attention]
related: [docs/patterns/attribution-patching.md, docs/patterns/activation-patching.md, docs/usage/backward-and-grad.md, docs/usage/source.md]
sources: [tests/test_atp_star.py]
---

# AtP* Building Blocks

> **Scope:** This page provides individually tested building blocks, not a
> drop-in end-to-end AtP* runner. The executable GPT-2/Pythia benchmark and
> Student-t confidence calculation are tracked as follow-up work in issue #704.
> Use the pieces below only after defining a model-specific component set,
> clean/noise direction, and scalar metric.

## What this is for

Attribution patching (AtP) cheaply ranks components using a first-order
approximation. AtP* addresses two ways that approximation can hide important
components:

- **Saturated attention softmax:** gradients through Q/K can be tiny even when
  patching Q or K would change the attention pattern substantially.
- **Residual cancellation:** direct and indirect gradient paths can cancel at a
  component even when either path has a large effect.

AtP* remains a screening method. Verify top candidates with exact activation
patching before making a causal claim.

## Cost summary

- Basic AtP needs a clean forward, a noise forward, and one backward pass.
- QK correction locally recomputes attention effects without another model pass.
- GradDrop uses one modified backward pass per residual layer instead of one
  ordinary backward pass.
- Exact verification still needs one patched forward per tested component.

## 1. Establish exact and AtP baselines

Capture clean activations, then noise activations and their metric gradients:

```python
with model.trace(clean_prompt):
    clean_activation = component.output.save()

with model.trace(noise_prompt):
    noise_ref = component.output
    noise_ref.requires_grad_(True)
    noise_activation = noise_ref.save()
    metric = metric_fn(model.lm_head.output[:, -1])
    noise_metric = metric.save()
    with metric.backward():
        noise_gradient = noise_ref.grad.clone().save()

atp = (clean_activation - noise_activation) * noise_gradient
```

Sum only over the dimensions inside one candidate component. For example, a
per-token residual score sums over hidden width but not sequence position.

For exact verification, repeat the noise trace and patch only one candidate:

```python
with model.trace(noise_prompt):
    component.output[:, token] = clean_activation[:, token]
    patched_metric = metric_fn(model.lm_head.output[:, -1]).save()

exact_effect = patched_metric - noise_metric
```

Use exact effects to compute Recall@K, rank correlation, and recovered causal
effect for the approximate ranking.

## 2. Access attention Q, K, V, and probabilities

With eager attention, GPT-2 and Pythia expose post-transform tensors through
`.source`. Transformers 4.x and 5.x use different operation names, argument
offsets, and output layouts, so resolve and normalize them explicitly:

```python
import torch

def attention_source_call(attention, family):
    names = {
        "gpt2": ("attention_interface_2", "attention_interface_0"),
        "pythia": ("attention_interface_2", "attention_interface_0", "unknown_0"),
    }
    if family not in names:
        raise ValueError(f"Unsupported AtP* model family: {family}")
    for name in names[family]:
        try:
            return getattr(attention.source, name)
        except AttributeError:
            pass
    raise ValueError(f"Unsupported {family} attention source layout")

def unpack_attention_call(call):
    args, _ = call.inputs
    offset = 0 if torch.is_tensor(args[0]) else 1
    query, key, value = args[offset : offset + 3]
    output, probabilities = call.output
    if output.shape == query.shape:
        normalized_output = output
    elif output.shape[-3:] == (
        query.shape[-2], query.shape[-3], query.shape[-1]
    ):
        normalized_output = output.transpose(-3, -2)
    else:
        raise ValueError("Unsupported per-head attention output layout")
    return query, key, value, normalized_output, probabilities

with model.trace(prompt):
    call = attention_source_call(attention, family)
    query, key, value, attention_output, probabilities = unpack_attention_call(call)
```

Use `model.transformer.h[layer].attn` for GPT-2 and
`model.gpt_neox.layers[layer].attention` for Pythia. The test suite covers both
legacy Transformers 4.48 and the current unified attention interface. Inspect
`attention.source` and fail explicitly for any unrecognized implementation.

## 3. Correct query attribution

Patch each clean query locally, recompute its softmax row against noise keys, and
compare the resulting per-head output with the noise output:

```python
def attention_probabilities(query, key, mask=None, scale=None):
    scores = torch.einsum("...qd,...kd->...qk", query, key)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    scores = scores * scale
    if mask is not None:
        scores = scores + mask
    return scores.softmax(dim=-1)

def query_output_delta(
    clean_query, noise_key, noise_value, noise_output, mask, scale=None
):
    probability = attention_probabilities(clean_query, noise_key, mask, scale)
    patched_output = torch.einsum("...qk,...kd->...qd", probability, noise_value)
    return patched_output - noise_output
```

Dot each local output delta with the noise-run gradient at the corresponding
per-head attention output. This keeps the exact local softmax change while
linearizing only the downstream network.

## 4. Correct key attribution in O(T²D)

Changing one key changes one logit in every applicable query row. The exact
local output delta can be computed without materializing every patched T-by-T
attention matrix:

```python
def key_output_delta(
    q, clean_k, noise_k, value, probability, output, scale=None
):
    delta = torch.einsum("...qd,...kd->...qk", q, clean_k - noise_k)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    delta = delta * scale

    log_odds = torch.log(probability) - torch.log1p(-probability)
    patched_p = torch.sigmoid(log_odds + delta)
    denominator = torch.where(
        probability == 1, torch.ones_like(probability), 1 - probability
    )
    probability_scale = (patched_p - probability) / denominator
    probability_scale = torch.where(
        probability == 1, torch.zeros_like(probability_scale), probability_scale
    )
    return probability_scale.unsqueeze(-1) * (
        value.unsqueeze(-3) - output.unsqueeze(-2)
    )
```

The result has shape `[batch, head, query, patched_key, head_dim]`. Dot it with
the per-head output gradient and sum over query and head-dimension to score each
key. `tests/test_atp_star.py` checks this formula against explicit single-key
patches, including a causal mask.

## 5. Detect residual cancellation with GradDrop

Run backward repeatedly from the same retained graph. On pass L, replace the
gradient entering residual contribution L with zero before reading upstream
component gradients:

```python
with metric.backward(retain_graph=True):
    ordinary_gradient = component_ref.grad.clone().save()

with metric.backward():
    residual_contribution.grad = torch.zeros_like(residual_contribution.grad)
    dropped_gradient = component_ref.grad.clone().save()
```

The residual contribution must execute downstream of the component being
scored so its gradient is encountered first during backward. Repeat for each
layer and aggregate exactly as specified by the AtP* paper. The test suite
includes a direct-plus-indirect cancellation model where ordinary AtP is zero
and the dropped gradient is nonzero.

For L residual layers, Equation 11 aggregates the per-layer estimates as:

```python
graddrop_score = drop_estimates.abs().sum(dim=0) / (num_layers - 1)
```

Compute this per clean/noise pair before averaging over the prompt distribution.
The L/(L-1) scaling preserves the direct path's expected contribution.

## 6. Verify and diagnose

1. Rank candidates with AtP*.
2. Patch the top K candidates individually and record exact effects.
3. On the unverified remainder, sample complementary Bernoulli subsets and
   patch each subset jointly.
4. Use the paper's paired subset statistics and Welch bound to report an upper
   confidence bound on a missed component's effect.

For each node, Algorithm 1's point estimate is:

```python
subset_estimate = abs(mean_effect_when_included - mean_effect_when_excluded)
```

Track included/excluded count, mean, and unbiased sample variance online for
every node. At least two samples are required in both groups. The confidence
diagnostic additionally needs a Student-t CDF; keep that dependency in the
benchmark environment rather than adding it to nnsight core.

Do not substitute a generic bootstrap or independent-node assumption for the
paper's diagnostic: interactions between jointly patched components are a stated
limitation.

## MVP benchmark protocol

Use the same prompt pairs, metric, component definitions, and verification
budget for every method:

- Models: `EleutherAI/pythia-70m-deduped` and `openai-community/gpt2`.
- Runtime: eager attention; float32 on CPU; record exact model revisions.
- Tasks: IOI-style name pairs and factual city completions.
- Methods: exact activation patching, AtP, AtP+QK, and AtP* with GradDrop.
- Quality: Recall@K, Spearman rank correlation, and exact effect recovered by
  the top-K ranking.
- Cost: model forward/backward equivalents, wall time, and peak memory.
- Reproducibility: seed, prompt pairs, scores, exact effects, and sampled masks.

Treat model-family comparisons as separate results if tokenization or component
counts differ. Do not pool ranks across incompatible node sets.

## Gotchas

- Force float32 for Pythia on CPU; checkpoint-default float16 can produce NaNs.
- Put every model in evaluation mode. Attention dropout invalidates the local
  probability identities used above.
- Q/K correction is exact only for the local attention-softmax change. The
  downstream score is still first-order.
- Treat masked probabilities of exactly zero and one explicitly without
  clamping valid small nonzero probabilities.
- Q, K, and V must use the same RoPE state, mask, and scaling as the model
  forward. Pass a non-default `scale` explicitly when the architecture does.
- Fused attention may not expose probabilities; use eager attention for the
  reference workflow rather than silently changing semantics.
- Access gradients in reverse forward order inside each backward session.

## Related

- [attribution-patching](attribution-patching.md)
- [activation-patching](activation-patching.md)
- [attention-patterns](attention-patterns.md)
- [backward-and-grad](../usage/backward-and-grad.md)
- [source](../usage/source.md)
- Kramár et al. (2024), [AtP*](https://arxiv.org/abs/2403.00745)

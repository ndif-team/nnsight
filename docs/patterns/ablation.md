---
title: Ablation
one_liner: Zero, mean, or noise the output of a component to measure its functional contribution.
tags: [pattern, interpretability, ablation, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/invoke-and-batching.md, docs/patterns/activation-patching.md, docs/patterns/per-head-attention.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracing/tracer.py]
---

# Ablation

## What this is for

Ablation removes (or replaces with a baseline) the output of a specific component and measures how the model's prediction degrades. It is the "lesion study" of mechanistic interpretability: if zeroing out head 5.4 destroys the model's IOI behavior, that head is doing something the task needs.

Common variants:

- **Zero ablation**: replace the component's output with zeros. Simple, but pushes the residual stream off-distribution.
- **Mean ablation**: replace with the average activation across some reference distribution. Stays closer to typical model state.
- **Noise / resampling ablation**: replace with a sample from another input. Useful when zero/mean is itself out-of-distribution.

In nnsight, ablation is just an in-place write inside a trace. Pair an ablated invoke with a baseline invoke (or an empty invoke) to compare.

## When to use

- Identifying which components a behavior depends on.
- Validating a circuit hypothesis: "if I delete every other head, does this circuit alone solve the task?"
- Measuring component importance for layer / head pruning.
- Sanity-checking probes: if you can read X off layer L but ablating L does not change the output, X may be epiphenomenal.

## Canonical pattern

Zero-ablate one MLP block, compare to baseline:

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
LAYER = 9

with model.trace() as tracer:
    # Baseline.
    with tracer.invoke(prompt):
        baseline_logits = model.lm_head.output[:, -1, :].save()

    # Ablated: zero out one MLP block.
    with tracer.invoke(prompt):
        model.transformer.h[LAYER].mlp.output[:] = 0
        ablated_logits = model.lm_head.output[:, -1, :].save()

paris = model.tokenizer.encode(" Paris")[0]
print(f"baseline P(Paris) = {baseline_logits.softmax(-1)[0, paris]:.3f}")
print(f"ablated  P(Paris) = {ablated_logits.softmax(-1)[0, paris]:.3f}")
```

Note the two invokes write to disjoint logits variables and never read from a shared module's output across invokes, so **no barrier is needed** here.

## Variations

### Zero a specific position only

```python
with model.trace(prompt):
    # Zero MLP output only at the last position.
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

For an attention output of shape `[batch, seq, n_heads * head_dim]`, reshape and zero one slice:

```python
import torch

n_heads, head_dim = model.config.n_head, model.config.n_embd // model.config.n_head

with model.trace(prompt):
    attn_out = model.transformer.h[LAYER].attn.output[0]  # [B, S, hidden]
    B, S, _ = attn_out.shape
    reshaped = attn_out.view(B, S, n_heads, head_dim).clone()
    reshaped[:, :, 4, :] = 0                              # zero head 4
    new_attn = reshaped.view(B, S, n_heads * head_dim)
    model.transformer.h[LAYER].attn.output = (new_attn,) + model.transformer.h[LAYER].attn.output[1:]
    logits = model.lm_head.output[:, -1, :].save()
```

For ergonomic per-head access, see `docs/patterns/per-head-attention.md`.

### Mean ablation

Compute the mean activation across a reference set in one session, then write it in:

```python
import torch
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ref_prompts = ["The capital of France is", "Paris is in", "Berlin lies in", "London hosts the"]

# Pass 1: collect mean MLP activations at LAYER, last position.
# Two patterns work for accumulating into a list — pick one:
# (1) Define `acts` OUTSIDE the trace and append in-place — no .save() on each element needed.
# (2) Define `acts = list().save()` INSIDE the trace and append normally.
LAYER = 9
acts = []                                    # pattern (1): list lives in caller frame
with model.trace() as tracer:
    for p in ref_prompts:
        with tracer.invoke(p):
            acts.append(model.transformer.h[LAYER].mlp.output[:, -1, :])  # no .save() needed

mean_act = torch.stack([a for a in acts], dim=0).mean(dim=0)  # [1, hidden]

# Pass 2: ablate with the mean.
prompt = "The Eiffel Tower is in the city of"
with model.trace(prompt):
    model.transformer.h[LAYER].mlp.output[:, -1, :] = mean_act
    logits = model.lm_head.output[:, -1, :].save()
```

### Noise / resampling ablation

Replace with the activation from a different (unrelated) prompt.

```python
clean   = "The Eiffel Tower is in the city of"
unrelated = "I went to the store and bought some"

with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke(unrelated):
        noise = model.transformer.h[LAYER].mlp.output[:, -1, :]
        barrier()
    with tracer.invoke(clean):
        barrier()
        model.transformer.h[LAYER].mlp.output[:, -1, :] = noise
        logits = model.lm_head.output[:, -1, :].save()
```

This is the inverse of activation patching: instead of "patch in clean", you "patch in noise". A barrier is required because both invokes touch the same module - see `docs/usage/barrier.md`.

### Empty invoke for batch-wide ablation

If you have multiple input invokes and want to ablate uniformly across the whole batch:

```python
with model.trace() as tracer:
    with tracer.invoke("Prompt A"):
        pass
    with tracer.invoke("Prompt B"):
        pass
    with tracer.invoke():                       # empty: full batch
        model.transformer.h[LAYER].mlp.output[:] = 0
        logits = model.lm_head.output[:, -1, :].save()  # shape [2, vocab]
```

Empty invokes are separate threads on the full batch and are useful when you want a single intervention site that sees every input. See `docs/usage/invoke-and-batching.md`.

## Interpretation tips

- **Compare logits, log-probs, or task accuracy** - not raw activation norms. The point is functional effect.
- **Zero ablation pushes out of distribution.** A large drop in P(answer) might just mean "anything missing here breaks the model", not "this specific component encodes the answer". Mean / resample ablation is the more conservative test.
- **One-component ablation can underestimate redundancy.** If two heads back each other up, zeroing either alone barely moves the answer. Try jointly ablating sets.
- **Measure direction, not just magnitude.** Sometimes ablation increases the wrong-answer probability without much changing the right-answer one - look at both.
- **Different ablation choices answer different questions.** Zero = "is there *any* signal here?" Mean = "is the *deviation from average* here important?" Resample = "is the *task-specific* content here important?"

## Gotchas

- `[:] = 0` is in-place. The same tensor reference downstream sees zeros. If you also need the pre-ablation value, `.clone().save()` it first.
- Submodule output shapes vary: `.attn.output` is a tuple `(attn_out, weights)` (across both `transformers<5.0` and `transformers>=5.0`), so index `[0]` for the attention output. `.mlp.output` is a tensor. Block outputs (`model.transformer.h[i].output`) are tuples in `transformers<5.0` and tensors in `transformers>=5.0`. Adjust `[0]` indexing and tuple-replacement (`(new,) + out[1:]`) accordingly. See `docs/usage/access-and-modify.md`.
- Accumulating activations into a Python list: either define the list **outside** the trace and append normally (in-place mutation flows to the caller frame), or define the list **inside** the trace and `.save()` it. Don't `.save()` every element individually — it's not needed.
- Module names differ across architectures. Use `print(model)` to inspect.
- If two invokes both read or write the same module's output, you need a `tracer.barrier(n)`. See `docs/usage/barrier.md`.

## Related

- [activation-patching](activation-patching.md) - The opposite operation: paste in a clean activation instead of zeroing.
- [attribution-patching](attribution-patching.md) - Linear approximation of patching that scales to whole-circuit sweeps.
- [per-head-attention](per-head-attention.md)
- [multi-prompt-comparison](multi-prompt-comparison.md)
- `docs/usage/access-and-modify.md`
- `docs/usage/barrier.md`

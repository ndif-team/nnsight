---
title: Activation Patching
one_liner: Replace activations from one run (e.g. clean) into another (e.g. corrupt) at a specific module to measure that component's causal contribution.
tags: [pattern, interpretability, causal-mediation, patching]
related: [docs/usage/barrier.md, docs/usage/invoke-and-batching.md, docs/patterns/attribution-patching.md, docs/patterns/multi-prompt-comparison.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/barrier.py]
---

# Activation Patching

## What this is for

Activation patching (a.k.a. causal mediation analysis, ROME-style "denoising")
asks: **does this component carry the information that determines the answer?** You
run two prompts:

- a **clean** prompt that produces the correct answer
- a **corrupt** prompt that produces a different answer

You take a single activation from the clean run and **paste it into the corrupt
run** at the corresponding module / position. If the corrupt run now predicts the
clean answer, that activation was sufficient to flip the model.

In nnsight you do this with two `tracer.invoke(...)` calls in one `trace()`. Because
both invokes touch the same module on the same forward, use a `tracer.barrier(n)` to
synchronize the value hand-off — see `docs/usage/barrier.md`.

Tutorial mirror: https://nnsight.net/notebooks/tutorials/activation_patching/

## When to use

- Localizing where a fact / behavior lives (which layer, which position, which head).
- Confirming that a candidate component (from a probe, attention pattern) is causal.
- Constructing causal traces over a (layer, position) grid.
- IOI, factual recall, and any task with a paired clean/corrupt design.

## Canonical pattern

Patch the residual stream at the **subject-token positions** from a clean prompt
into a corrupt one. (Equal-length prompts let you patch matching positions.)

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

clean   = "The Eiffel Tower is in the city of"   # next token: " Paris"
corrupt = "The Colosseum is in the city of"      # both prompts are 10 tokens

# NB: GPT-2 small is near-degenerate on the corrupt prompt -- its top-5 spans 0.3
# logits and " Rome" only ranks 3rd (' P', ' T', ' C', ' Rome', ' B'). Compare the
# *logit difference* below, not the top token, or small numerical differences
# between runs will flip the argmax and look like a bug.

paris = model.tokenizer.encode(" Paris")[0]
rome  = model.tokenizer.encode(" Rome")[0]

LAYER = 0
SUBJECT = slice(1, 5)          # the subject tokens differ between the two prompts

with model.trace() as tracer:
    barrier = tracer.barrier(2)

    # Invoke 1: capture the clean residual stream at LAYER.
    with tracer.invoke(clean):
        clean_hs = model.transformer.h[LAYER].output
        barrier()                       # signal: clean_hs is ready

    # Invoke 2: corrupt run, with the clean subject activations patched in.
    with tracer.invoke(corrupt):
        barrier()                       # wait until clean_hs is materialized
        hs = model.transformer.h[LAYER].output
        hs[:, SUBJECT, :] = clean_hs[:, SUBJECT, :]
        model.transformer.h[LAYER].output = hs
        patched_logits = model.lm_head.output[:, -1, :].save()

    # Invoke 3: corrupt-only baseline, no patching.
    with tracer.invoke(corrupt):
        baseline_logits = model.lm_head.output[:, -1, :].save()

pb, pp = baseline_logits.softmax(-1), patched_logits.softmax(-1)
print(f"baseline corrupt: P(Paris)={pb[0,paris]:.3f} P(Rome)={pb[0,rome]:.3f}")
print(f"patched  corrupt: P(Paris)={pp[0,paris]:.3f} P(Rome)={pp[0,rome]:.3f}")
```

Real output — patching the subject token at layer 0 pushes the corrupt run toward
the clean answer:

```
baseline corrupt: P(Paris)=0.003 P(Rome)=0.014
patched  corrupt: P(Paris)=0.064 P(Rome)=0.006
```

**Why the barrier?** Both invokes access `model.transformer.h[LAYER].output`.
Without it, invoke 2 tries to use `clean_hs` before invoke 1 has produced it — a
`NameError`. The barrier parks invoke 1 *after* it reads `clean_hs` and invoke 2
*before* it writes, so the value crosses cleanly. See `docs/usage/barrier.md`.

**The reassignment is optional here.** Reading `.output` returns the live tensor,
so `hs[:, SUBJECT, :] = ...` has already propagated; `model...output = hs` just
makes the intent explicit. You need a real assignment only when you have a
*different* tensor to put in place of the old one — a reshape, a stack, an
arithmetic result. For a tuple `.output`, assign the rebuilt tuple
(`module.output = (new,) + tuple(out[1:])`), since a tuple's elements cannot be
reassigned individually.

## Variations

### Sweep over layers

```python
results = {}
for layer in range(len(model.transformer.h)):
    with model.trace() as tracer:
        barrier = tracer.barrier(2)
        with tracer.invoke(clean):
            hs_clean = model.transformer.h[layer].output
            barrier()
        with tracer.invoke(corrupt):
            barrier()
            hs = model.transformer.h[layer].output
            hs[:, SUBJECT, :] = hs_clean[:, SUBJECT, :]
            model.transformer.h[layer].output = hs
            logits = model.lm_head.output[:, -1, :].save()
    results[layer] = logits.softmax(-1)[0, paris].item()

for layer, p in results.items():
    print(f"layer {layer:2d}: P(Paris)={p:.3f}")
```

```
layer  0: P(Paris)=0.064
layer  1: P(Paris)=0.057
layer  2: P(Paris)=0.051
layer  4: P(Paris)=0.040
layer  6: P(Paris)=0.023
layer  8: P(Paris)=0.012
layer 10: P(Paris)=0.004
```

The city information at the subject token is read out in the **early** layers —
patching there flips the prediction, patching late layers does almost nothing.

### Patch the last position instead

`SUBJECT = slice(-1, None)` (or `[:, -1, :]`) answers "is the *final* prediction
sensitive to this layer?" On GPT-2 this barely moves the answer until the last
layer — the fact is retrieved at the subject token, not the final one. Position
choice changes the question you're asking.

### Patch attention output / MLP output instead of residual

```python
# Attention output of block LAYER (a tuple -> element 0 is the tensor)
clean_attn = model.transformer.h[LAYER].attn.output[0][:, -1, :]
# ...
out = model.transformer.h[LAYER].attn.output
new = out[0]
new[:, -1, :] = clean_attn
model.transformer.h[LAYER].attn.output = (new,) + tuple(out[1:])

# MLP output of block LAYER (a plain tensor)
clean_mlp = model.transformer.h[LAYER].mlp.output[:, -1, :]
# ...
mlp = model.transformer.h[LAYER].mlp.output
mlp[:, -1, :] = clean_mlp
model.transformer.h[LAYER].mlp.output = mlp
```

Check tuple-vs-tensor with `isinstance(module.output, tuple)` (or `model.scan`)
before indexing — see `docs/usage/access-and-modify.md`.

### Per-head patching

Reshape the attention output to `[batch, seq, n_heads, head_dim]` and patch a
single head. See `docs/patterns/per-head-attention.md`.

### Noising direction (corrupt the clean run)

Start from a clean run and *inject* a corrupt activation. The pattern is symmetric —
swap the prompts and invoke order.

## Interpretation tips

- **Always run a no-patch baseline** for the corrupt prompt in the same `trace()`.
  Tokenizer / batch effects shift logits slightly; the baseline is the reference.
- **Patching the residual at layer L is cumulative** — it overwrites everything up
  to and including L. To isolate one component, patch the sub-block (`.attn` /
  `.mlp`) or use attribution patching.
- **Position is critical.** Patching the subject token localizes the *source* of a
  fact; patching the last position tests *sensitivity* of the final prediction.
- **Effects can be small.** A few-percent shift on a sharp prompt is often real.
  Average over several prompt pairs.
- **Barrier count = number of invokes that call `barrier()`.** A third no-patch
  invoke that doesn't synchronize is not counted.

## Gotchas

- **The barrier is needed because invoke 2 *uses* a value from invoke 1.** If both
  invokes only read their own activations (no value flowing between them), no
  barrier is needed.
- Within one invoke, modules must be accessed in forward-pass order, or you hit
  `OutOfOrderError`. You cannot capture `h[5]` after `h[10]` in the same invoke.
- Clean and corrupt prompts of **different token lengths** can't share a position
  slice. Match lengths, or patch a common suffix / the last position.
- Block outputs are plain tensors in current `transformers`; attention outputs are
  tuples. Adjust `[0]` indexing accordingly. See `docs/usage/access-and-modify.md`.
- Patching mutates the running tensor. `.clone().save()` first if you also want the
  unmodified state.

## Related

- [attribution-patching](attribution-patching.md) — linear approximation that gets a
  whole causal map from one clean + one corrupt run.
- [multi-prompt-comparison](multi-prompt-comparison.md)
- [per-head-attention](per-head-attention.md)
- `docs/usage/barrier.md`
- `docs/usage/invoke-and-batching.md`
- https://nnsight.net/notebooks/tutorials/activation_patching/
- Meng et al. (2022), "Locating and Editing Factual Associations in GPT" (ROME).
- Wang et al. (2022), "Interpretability in the Wild" (IOI).

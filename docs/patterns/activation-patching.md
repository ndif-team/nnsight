---
title: Activation Patching
one_liner: Replace activations from one run (e.g. clean) into another (e.g. corrupt) at a specific module to measure that component's causal contribution.
tags: [pattern, interpretability, causal-mediation, patching]
related: [docs/usage/barrier.md, docs/usage/invoke-and-batching.md, docs/patterns/attribution-patching.md, docs/patterns/multi-prompt-comparison.md]
sources: [src/nnsight/intervention/tracing/tracer.py:551, src/nnsight/intervention/envoy.py]
---

# Activation Patching

## What this is for

Activation patching (a.k.a. causal mediation analysis, ROME-style "denoising") asks: **does this component carry the information that determines the answer?** You run two prompts through the model:

- a **clean** prompt that produces the correct answer
- a **corrupt** prompt that produces a wrong / different answer

You then take a single activation from the clean run and **paste it into the corrupt run** at the corresponding module / position. If the corrupt run now predicts the clean answer, that activation was sufficient to flip the model. If the prediction barely moves, that activation does not carry the relevant information.

In nnsight you do this with two `tracer.invoke(...)` calls in the same `trace()`. Because both invokes touch the same module on the same forward iteration, you must use a `tracer.barrier(n)` to synchronize the variable hand-off between them - see `docs/usage/barrier.md`.

Tutorial mirror: https://nnsight.net/notebooks/tutorials/activation_patching/

## When to use

- Localizing where a fact / behavior lives in a model (which layer, which position, which head).
- Confirming that a candidate component identified by another method (probe, attention pattern) is actually causal.
- Constructing causal traces over a (layer, position) grid.
- IOI, indirect-object-identification, factual recall, and any task where you have a paired clean/corrupt design.

## Canonical pattern

Patch the residual stream at one layer, at the last token position, from a clean prompt into a corrupt prompt:

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

clean   = "The Eiffel Tower is in the city of"   # next token: " Paris"
corrupt = "The Colosseum is in the city of"      # next token: " Rome"

LAYER = 7

with model.trace() as tracer:
    barrier = tracer.barrier(2)

    # Invoke 1: capture clean residual stream at the last token of LAYER.
    with tracer.invoke(clean):
        clean_hs = model.transformer.h[LAYER].output[0][:, -1, :]
        barrier()                       # signal: clean_hs is ready

    # Invoke 2: corrupt run, with the clean activation patched in.
    with tracer.invoke(corrupt):
        barrier()                       # wait until clean_hs is materialized
        model.transformer.h[LAYER].output[0][:, -1, :] = clean_hs
        patched_logits = model.lm_head.output[:, -1, :].save()

    # Invoke 3: corrupt-only baseline (for comparison), no patching.
    with tracer.invoke(corrupt):
        baseline_logits = model.lm_head.output[:, -1, :].save()

paris = model.tokenizer.encode(" Paris")[0]
rome  = model.tokenizer.encode(" Rome")[0]

print(f"baseline corrupt: P(Paris)={baseline_logits.softmax(-1)[0, paris]:.3f}, P(Rome)={baseline_logits.softmax(-1)[0, rome]:.3f}")
print(f"patched corrupt : P(Paris)={patched_logits.softmax(-1)[0, paris]:.3f}, P(Rome)={patched_logits.softmax(-1)[0, rome]:.3f}")
```

If layer 7 carries the city information, P(Paris) should rise and P(Rome) should fall in the patched run.

**Why the barrier?** Both invokes access `model.transformer.h[LAYER].output`. Without a barrier, invoke 2 starts before invoke 1 has produced `clean_hs`, and you get a `NameError`. The barrier blocks invoke 1 *after* it captures `clean_hs` and blocks invoke 2 *before* it tries to use it, ensuring the value crosses cleanly. See `docs/usage/barrier.md`.

## Variations

### Sweep over layers

The pattern above patches one layer. To build a causal map, sweep:

```python
results = {}
for layer in range(len(model.transformer.h)):
    with model.trace() as tracer:
        barrier = tracer.barrier(2)
        with tracer.invoke(clean):
            hs = model.transformer.h[layer].output[0][:, -1, :]
            barrier()
        with tracer.invoke(corrupt):
            barrier()
            model.transformer.h[layer].output[0][:, -1, :] = hs
            logits = model.lm_head.output[:, -1, :].save()
    results[layer] = logits.softmax(-1)[0, paris].item()
```

Plot `results` to see at which layer patching the residual stream most increases P(clean answer).

### Patch a specific position (token)

Replace `[:, -1, :]` with `[:, pos, :]` to patch only one token. For factual recall, the subject token (e.g. " Eiffel") is often the relevant position.

### Patch attention output / MLP output instead of residual

```python
# Attention output of block LAYER
clean_attn = model.transformer.h[LAYER].attn.output[0][:, -1, :]
# ...
model.transformer.h[LAYER].attn.output[0][:, -1, :] = clean_attn

# MLP output of block LAYER
clean_mlp = model.transformer.h[LAYER].mlp.output[:, -1, :]
# ...
model.transformer.h[LAYER].mlp.output[:, -1, :] = clean_mlp
```

The shape and tuple-vs-tensor of `.output` varies by submodule type - check with `model.scan(prompt)` first if unsure.

### Per-head patching

For attention output, reshape to `[batch, seq, n_heads, head_dim]` and patch a single head. See `docs/patterns/per-head-attention.md`.

### Noising direction (corrupt the clean run)

Some setups go the other way: start from a clean run and *inject* a corrupt activation. The pattern is symmetric - swap the prompts and invoke order.

## Interpretation tips

- **Always run a no-patch baseline** for the corrupt prompt in the same `trace()`. Tokenizer / batch effects can shift logits slightly; the baseline tells you the natural answer rate.
- **Patching the residual at layer L is cumulative.** It overwrites everything up to and including layer L. If you only want to know the *contribution* of one component, patch the sub-block (`.attn` or `.mlp`) or use attribution patching.
- **Position is critical.** Patching the last position is what answers "is the prediction sensitive to this layer here?" but for the *source* of information, patching the subject token is usually more revealing.
- **Effects can be small.** A 5-10% probability shift on a sharp prompt is often a real and replicable signal. Run several prompt pairs and average.
- **Barrier count = number of invokes that call `barrier()`**. If you have a third no-patch invoke that does not synchronize, do not include it in the count.

## Gotchas

- Both invokes touch `transformer.h[L].output` - you **must** use `tracer.barrier(2)`. Without it, `clean_hs` is undefined when invoke 2 runs. See `docs/usage/barrier.md`.
- Inside one invoke, modules must be accessed in forward-pass order. You cannot capture `h[5]` after `h[10]` in the same invoke.
- `block.output[0]` is the residual stream; `block.output` is the full tuple.
- Patching modifies the running tensor in place. Save a `.clone()` first if you also want the unmodified post-patch state.
- See `docs/usage/access-and-modify.md` for in-place vs replacement semantics on tuple outputs.

## Related

- [attribution-patching](attribution-patching.md) - Linear approximation that gets you an entire causal map from one clean and one corrupt run.
- [multi-prompt-comparison](multi-prompt-comparison.md)
- [per-head-attention](per-head-attention.md)
- `docs/usage/barrier.md`
- `docs/usage/invoke-and-batching.md`
- https://nnsight.net/notebooks/tutorials/activation_patching/
- Meng et al. (2022), "Locating and Editing Factual Associations in GPT" (ROME).
- Wang et al. (2022), "Interpretability in the Wild" (IOI).

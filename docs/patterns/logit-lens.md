---
title: Logit Lens
one_liner: Apply the final layer norm and unembedding to intermediate hidden states to read off what each layer "thinks" the next token is.
tags: [pattern, interpretability, residual-stream, decoding]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/patterns/multi-prompt-comparison.md]
sources: [src/nnsight/intervention/envoy.py:239, src/nnsight/modeling/language.py]
---

# Logit Lens

## What this is for

The logit lens (nostalgebraist, 2020) reads the residual stream at every transformer layer through the model's own final layer norm and unembedding (`lm_head`). The result is a token-distribution-per-layer: a way to ask "if the model stopped thinking right now, what would it predict?"

In a forward pass of a decoder-only transformer, the residual stream at layer L is `h_L`. The model's final prediction is `lm_head(ln_f(h_final))`. The logit lens applies the same head to earlier layers: `lm_head(ln_f(h_L))` for L = 0, 1, 2, ... This often shows a smooth refinement: early layers predict bag-of-words frequent tokens, middle layers track syntactic role, late layers converge on the actual answer.

In nnsight, you can call `model.lm_head(...)` directly inside a trace. When you call a wrapped module like a function inside a trace, it dispatches to `forward()` instead of `__call__()`, which **bypasses the interleaving hooks** (see `src/nnsight/intervention/envoy.py:239`). That means the call runs without registering a `.input`/`.output` event and without running a second pass through the model - it is just the linear math you want.

Tutorial mirror: https://nnsight.net/notebooks/tutorials/logit_lens/

## When to use

- Visualizing layer-wise prediction trajectories on a single prompt.
- Locating the layer at which a specific fact / token becomes the top-1 prediction.
- Sanity-checking that a model "knows" something before doing more invasive interventions.
- Pairing with activation patching to ask "at what layer does patching this token's prediction help?"

## Canonical pattern

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    # Apply final ln + unembedding to every block's residual output.
    layer_top_tokens = []
    for block in model.transformer.h:
        hs = block.output[0]                    # residual stream at this layer
        logits = model.lm_head(model.transformer.ln_f(hs))
        top_tok = logits[:, -1, :].argmax(dim=-1).save()
        layer_top_tokens.append(top_tok)

for i, tok in enumerate(layer_top_tokens):
    print(f"layer {i:2d}: {model.tokenizer.decode(tok[0])!r}")
```

You should see early layers predict generic tokens and the final layer converge on " Paris".

## Variations

### Top-k per layer

```python
with model.trace(prompt):
    per_layer_topk = []
    for block in model.transformer.h:
        hs = block.output[0]
        logits = model.lm_head(model.transformer.ln_f(hs))
        topk = logits[:, -1, :].topk(5, dim=-1).indices.save()
        per_layer_topk.append(topk)

for i, topk in enumerate(per_layer_topk):
    decoded = [model.tokenizer.decode(t) for t in topk[0]]
    print(f"layer {i:2d}: {decoded}")
```

### Probability of a target token across layers

```python
import torch

target = " Paris"
target_id = model.tokenizer.encode(target)[0]

with model.trace(prompt):
    target_probs = []
    for block in model.transformer.h:
        hs = block.output[0]
        logits = model.lm_head(model.transformer.ln_f(hs))
        prob = logits[:, -1, :].softmax(dim=-1)[:, target_id].save()
        target_probs.append(prob)

for i, p in enumerate(target_probs):
    print(f"layer {i:2d}: P({target!r}) = {p.item():.3f}")
```

### Tuned lens (use a learned linear map per layer)

If you have a tuned-lens checkpoint with one affine map `A_L` per layer, replace `model.transformer.ln_f(hs)` with `A_L(hs)`:

```python
# tuned_maps: list of nn.Linear, one per layer, on model.device
with model.trace(prompt):
    per_layer = []
    for L, block in enumerate(model.transformer.h):
        hs = block.output[0]
        logits = model.lm_head(tuned_maps[L](hs))
        per_layer.append(logits[:, -1, :].argmax(dim=-1).save())
```

### MLP-only / attention-only lens

Some research splits the residual into the attention contribution vs the MLP contribution. Access the sub-block outputs directly (`block.attn.output[0]` and `block.mlp.output`) and project those.

## Interpretation tips

- **Look at the layer where the answer first becomes top-1.** That layer is doing the bulk of the "decision". Layers after it are usually refinement.
- **Diverging top-1s** between adjacent layers signal a competing hypothesis - useful for finding ambiguity.
- **Probability, not just argmax.** Argmax can hide a 0.51 vs 0.49 race. Soft-max probabilities or log-probs over a target are usually more informative.
- **Position matters.** `[:, -1, :]` reads the last position (next-token prediction). For factual recall tasks, the relevant position is often the subject token, not the final token.
- **Layer norm matters.** Skipping `ln_f` gives garbage results - the unembedding expects normalized inputs. Use the model's own `ln_f`, not a fresh `LayerNorm`.
- **Model-specific module names.** GPT-2 uses `model.transformer.h[i]` and `model.transformer.ln_f`. Llama / Mistral / Qwen typically use `model.model.layers[i]` and `model.model.norm`. Use `print(model)` to inspect.

## Gotchas

- `block.output` shape varies by transformers version. **In `transformers<5.0` it's a tuple** `(residual, present, attentions, ...)` and you index `[0]` to get the residual stream. **In `transformers>=5.0` block outputs are no longer tuples** — `block.output` *is* the residual tensor directly. If `block.output[0]` looks wrong on your model, check your transformers version and drop the `[0]`. See `docs/usage/access-and-modify.md`.
- Calling `model.lm_head(...)` inside a trace runs `forward()`, not `__call__()`, which is what you want here. Calling it via something that triggers the interleaving system (e.g. `model.lm_head.output`) would mean "intercept the lm_head call that the model itself makes", which is a different operation.
- Do not save inside a Python list and then expect `print(layer_top_tokens)` to show tensors after the trace if you forgot `.save()` on each element. Each tensor needs `.save()` (or wrap the list with `nnsight.save(...)`).

## Related

- [activation-patching](activation-patching.md) - Pair logit lens with patching to localize where a fact lives.
- [attention-patterns](attention-patterns.md)
- [gradient-based-attribution](gradient-based-attribution.md)
- https://nnsight.net/notebooks/tutorials/logit_lens/
- nostalgebraist (2020), "interpreting GPT: the logit lens".

---
title: Logit Lens
one_liner: Apply the final layer norm and unembedding to intermediate hidden states to read off what each layer "thinks" the next token is.
tags: [pattern, interpretability, residual-stream, decoding]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/patterns/activation-patching.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/modeling/transformers.py]
---

# Logit Lens

## What this is for

The logit lens (nostalgebraist, 2020) reads the residual stream at every
transformer layer through the model's own final layer norm and unembedding
(`lm_head`). The result is a token-distribution-per-layer: a way to ask "if the
model stopped thinking right now, what would it predict?"

The residual stream at layer L is `h_L`. The model's final prediction is
`lm_head(ln_f(h_final))`. The logit lens applies the same head to earlier layers:
`lm_head(ln_f(h_L))` for L = 0, 1, 2, ... This often shows a smooth refinement:
early layers predict generic frequent tokens, late layers converge on the answer.

Calling `model.lm_head(...)` inside a trace runs `forward()` directly and
**bypasses the interleaving hooks** — it is just the linear math you want, applied
out of order without re-triggering the model. See
`docs/usage/access-and-modify.md` ("Calling modules directly inside a trace").

Tutorial mirror: https://nnsight.net/notebooks/tutorials/logit_lens/

## When to use

- Visualizing layer-wise prediction trajectories on a single prompt.
- Locating the layer at which a specific fact / token becomes the top-1 prediction.
- Sanity-checking that a model "knows" something before doing more invasive
  interventions.
- Pairing with activation patching to ask "at what layer does patching this token's
  prediction help?"

## Canonical pattern

```python
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    # Apply final ln + unembedding to every block's residual output.
    layer_top_tokens = nnsight.save([])
    for block in model.transformer.h:
        hs = block.output                    # residual stream at this layer (a Tensor)
        logits = model.lm_head(model.transformer.ln_f(hs))
        top_tok = logits[:, -1, :].argmax(dim=-1)
        layer_top_tokens.append(top_tok)

for i, tok in enumerate(layer_top_tokens):
    print(f"layer {i:2d}: {model.tokenizer.decode(tok[0])!r}")
```

Real output on GPT-2 — the answer emerges at layer 10:

```
layer  0: ' the'
layer  1: ' the'
layer  2: ' the'
layer  3: ' the'
layer  4: ' the'
layer  5: ' the'
layer  6: ' East'
layer  7: ' Ing'
layer  8: ' Rome'
layer  9: ' London'
layer 10: ' Paris'
layer 11: ' Paris'
```

In current `transformers`, a GPT-2 block's `.output` is a plain
`Tensor` `(batch, seq, hidden)` — read `block.output` directly, **not**
`block.output[0]`.

## Variations

### Top-k per layer

```python
with model.trace(prompt):
    per_layer_topk = nnsight.save([])
    for block in model.transformer.h:
        hs = block.output
        logits = model.lm_head(model.transformer.ln_f(hs))
        topk = logits[:, -1, :].topk(5, dim=-1).indices
        per_layer_topk.append(topk)

for i, topk in enumerate(per_layer_topk):
    decoded = [model.tokenizer.decode(t) for t in topk[0]]
    print(f"layer {i:2d}: {decoded}")
```

```
layer  8: [' Rome', ' London', ' Chicago', ' San', ' La']
layer  9: [' London', ' Paris', ' Amsterdam', ' Rome', ' Chicago']
layer 10: [' Paris', ' London', ' Amsterdam', ' Berlin', ' Hamburg']
layer 11: [' Paris', ' London', ' New', ' Amsterdam', ' Berlin']
```

### Probability of a target token across layers

```python
target = " Paris"
target_id = model.tokenizer.encode(target)[0]

with model.trace(prompt):
    target_probs = nnsight.save([])
    for block in model.transformer.h:
        hs = block.output
        logits = model.lm_head(model.transformer.ln_f(hs))
        prob = logits[:, -1, :].softmax(dim=-1)[:, target_id]
        target_probs.append(prob)

for i, p in enumerate(target_probs):
    print(f"layer {i:2d}: P({target!r}) = {p.item():.3f}")
```

```
layer  7: P(' Paris') = 0.003
layer  8: P(' Paris') = 0.025
layer  9: P(' Paris') = 0.248
layer 10: P(' Paris') = 0.183
layer 11: P(' Paris') = 0.070
```

The probability peaks mid-late (layer 9) then settles — argmax alone would hide
that.

### Tuned lens (learned linear map per layer)

If you have a tuned-lens checkpoint with one affine map `A_L` per layer, replace
`model.transformer.ln_f(hs)` with `A_L(hs)`:

```python
# tuned_maps: list of nn.Linear, one per layer, on model.device
with model.trace(prompt):
    per_layer = nnsight.save([])
    for L, block in enumerate(model.transformer.h):
        hs = block.output
        logits = model.lm_head(tuned_maps[L](hs))
        per_layer.append(logits[:, -1, :].argmax(dim=-1))
```

### MLP-only / attention-only lens

To project the attention or MLP contribution separately, read the sub-block
outputs directly: `block.mlp.output` (a Tensor) and `block.attn.output[0]` (the
first element of the attention tuple).

## Interpretation tips

- **Look at the layer where the answer first becomes top-1.** That layer does the
  bulk of the decision; later layers usually refine.
- **Probability, not just argmax.** Argmax hides a 0.51 vs 0.49 race. Softmax
  probabilities over a target are more informative.
- **Position matters.** `[:, -1, :]` reads the last position (next-token
  prediction). For factual recall the relevant position is often the subject token.
- **Layer norm matters.** Skipping `ln_f` gives garbage — the unembedding expects
  normalized inputs. Use the model's own `ln_f`.
- **Model-specific module names.** GPT-2 uses `model.transformer.h[i]` and
  `model.transformer.ln_f`. Llama / Mistral / Qwen typically use
  `model.model.layers[i]` and `model.model.norm`. Use `print(model)` to inspect.

## Gotchas

- In current `transformers`, GPT-2 **block** outputs are plain tensors —
  `block.output` *is* the residual tensor. Older `transformers<5.0` returned a
  tuple; if `block.output[0]` looks wrong, drop the `[0]`. See
  `docs/usage/access-and-modify.md`.
- Calling `model.lm_head(...)` inside a trace runs `forward()` (no hooks), which is
  what you want. Reading `model.lm_head.output` instead would intercept the *real*
  `lm_head` call the model itself makes — a different operation.
- To collect values into a list, create the list inside the trace with
  `xs = nnsight.save([])` and append raw values (no per-element `.save()`).
  Without the saved, named list the values aren't available after the trace.

## Related

- [activation-patching](activation-patching.md) — pair logit lens with patching to
  localize where a fact lives.
- [attention-patterns](attention-patterns.md)
- `docs/usage/access-and-modify.md`
- https://nnsight.net/notebooks/tutorials/logit_lens/
- nostalgebraist (2020), "interpreting GPT: the logit lens".

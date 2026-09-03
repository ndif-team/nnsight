---
title: Logit Lens
one_liner: Apply the final layer norm and unembedding to intermediate hidden states to read off what each layer "thinks" the next token is.
tags: [pattern, interpretability, residual-stream, decoding]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/patterns/activation-patching.md, docs/patterns/probing.md]
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

On `VLLM`, `model.lm_head(h)` raises (`LMHead's weights should be used in the sampler`); the
lens is `model.logits_processor(model.lm_head, model.model.norm(h))` on `h = (out[0] + out[1])[-1:]`
— see [Tensor parallelism](../models/vllm-parallelism.md#tensor-parallelism-is-transparent).

Calling `model.lm_head(...)` inside a trace runs the module with the trace
**stood down** — it is just the linear math you want, applied out of order
without re-triggering the model. See `docs/usage/access-and-modify.md`
("Calling modules directly inside a trace").

Some architectures post-process the head's output, so `model.lm_head.output` is not
always the model's logits. One line tells you whether yours is one of them — see
[Check the wiring](#check-the-wiring) below, and run it before you plot anything.

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

decoded = [model.tokenizer.decode(tok[0]) for tok in layer_top_tokens]
for i, tok in enumerate(decoded):
    print(f"layer {i:2d}: {tok!r}")

assert decoded[:6] == [" the"] * 6
assert decoded[-2:] == [" Paris", " Paris"]
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

## Check the wiring

Applied to the *last* block, the lens is the model's own final computation. So it
has to reproduce the model's own logits exactly, and one line says whether the
norm, the head and the block output you picked are the right three:

```python
import torch

with model.trace(prompt):
    lens = model.lm_head(model.transformer.ln_f(model.transformer.h[-1].output)).save()
    real = model.output.logits.save()

assert torch.equal(lens, real), (lens - real).abs().max()
```

`True` on GPT-2, SmolLM2-135M, Qwen2.5-0.5B and pythia-70m (nnsight 0.8,
transformers 5.15). A `False` means one of three things, all of which produce a
plot rather than an error if you skip this:

- the wrong norm (or none — see the note under [Interpretation tips](#interpretation-tips)),
- the wrong head (`model.embed_out` does not exist on pythia; it is `model.lm_head`),
- or the model post-processes the head's output.

The third case is Gemma-2, which applies `tanh` logit softcapping in
`Gemma2ForCausalLM.forward` rather than inside `lm_head`:

```python
logits = self.lm_head(hidden_states[:, slice_indices, :])
if self.config.final_logit_softcapping is not None:
    logits = logits / self.config.final_logit_softcapping
    logits = torch.tanh(logits)
    logits = logits * self.config.final_logit_softcapping
```

On `google/gemma-2-2b` in float32, `final_logit_softcapping = 30.0`, the check
above fails by `51.0`, and the uncapped distribution is far too confident: max
probability `0.9995` against the model's own `0.9257`, entropy `0.0050` against
`0.6010` — a factor of 120. The top-1 token usually survives; nothing else does.

Apply the same cap yourself and intermediate layers are read on the model's own
scale:

```python
cap = getattr(model.config, "final_logit_softcapping", None)
logits = model.lm_head(model.model.norm(hs))
if cap is not None:
    logits = torch.tanh(logits / cap) * cap
```

That is bit-identical to `model.output.logits` at the last layer. Check
`model.config` rather than a list of model names: Gemma-3 sets
`final_logit_softcapping` to `None`.

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

### Token labels for heatmaps

If you plot per-token logit lens values, tokenize the prompt outside the trace to
build x-axis labels. This keeps the plotting code independent of trace internals:

```python
token_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
token_labels = [model.tokenizer.decode([int(token_id)]) for token_id in token_ids]
```

Assume `layer_token_scores` is a `[layers, sequence]` array of values to plot.
Use numeric x positions for Plotly heatmaps, then display the token strings as
tick labels. That keeps repeated tokens in separate columns:

```python
import plotly.express as px

x_positions = list(range(len(token_labels)))

fig = px.imshow(
    layer_token_scores,
    x=x_positions,
    y=list(range(layer_token_scores.shape[0])),
    labels={"x": "input token", "y": "layer", "color": "score"},
)
fig.update_xaxes(tickmode="array", tickvals=x_positions, ticktext=token_labels)
fig.show()
```

### Tuned lens (learned translator per layer)

A tuned lens keeps the model's frozen final norm and unembedding and learns one
affine *translator* `A_L` that maps layer `L`'s residual into the final layer's
basis first ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112)). Only the
translator is new; the norm stays:

```python
# tuned_maps: list of nn.Linear, one per layer, on model.device
with model.trace(prompt):
    per_layer = nnsight.save([])
    for L, block in enumerate(model.transformer.h):
        hs = block.output
        translated = tuned_maps[L](hs)               # affine, initialized to the identity
        logits = model.lm_head(model.transformer.ln_f(translated))
        per_layer.append(logits[:, -1, :].argmax(dim=-1))
```

Dropping `ln_f` and decoding `lm_head(A_L(hs))` instead is not the same model and
not a free reparameterization: a LayerNorm is not affine, and its per-token
normalization scale varies by a factor of 33 across the positions of one GPT-2
prompt. The best affine fit to `ln_f` on GPT-2 layer 8 leaves a relative residual
of 0.42 on natural text.

### MLP-only / attention-only lens

To project the attention or MLP contribution separately, read the sub-block
outputs directly: `block.mlp.output` (a Tensor) and `block.attn.output[0]` (the
first element of the attention tuple).

## Interpretation tips

- **Look at the layer where the answer first becomes top-1.** That layer does the
  bulk of the decision; later layers usually refine.
- **GPT-2's smooth trajectory is not the general case.** On Qwen2.5-0.5B
  "The capital of France is" decodes to punctuation and code fragments for the
  first 21 of 24 layers, then `' Paris'` for the last three. SmolLM2-135M gives
  punctuation for its first half, `' the'` through most of the second, and
  `' Paris'` only at layer 29 of 30. Both models answer correctly. Once the
  wiring check passes, a flat curve is as often a fact about the lens as about
  the model.
- **Probability, not just argmax.** Argmax hides a 0.51 vs 0.49 race. Softmax
  probabilities over a target are more informative.
- **Position matters.** `[:, -1, :]` reads the last position (next-token
  prediction). For factual recall the relevant position is often the subject token.
- **Layer norm matters, and skipping it does not look like an error.** Without
  `ln_f`, GPT-2 decodes `' the'` at *every* layer with probability `1.0000` — a
  lens that looks more confident than the correct one, and identical to the
  symptom that is supposed to send you off to fit a tuned lens. Check the wiring
  before concluding that the lens does not transfer.
- **Model-specific module names.** GPT-2 uses `model.transformer.h[i]` and
  `model.transformer.ln_f`. Llama / Mistral / Qwen typically use
  `model.model.layers[i]` and `model.model.norm`. Use `print(model)` to inspect.

## Gotchas

- In current `transformers`, a **block**'s output is a plain tensor on every
  family tested — GPT-2, Llama, Qwen, GPT-NeoX, Gemma-2. `block.output` *is* the
  residual tensor, and `block.output[0]` indexes the batch. (`transformers < 5.0`
  wraps it in a tuple, which is where the `[0]` in older code comes from.) See
  `docs/usage/access-and-modify.md`.
- Calling `model.lm_head(...)` inside a trace runs `forward()` with the trace stood
  down, which is what you want. Reading `model.lm_head.output` instead would
  intercept the *real* `lm_head` call the model itself makes — a different
  operation.

## Related

- [activation-patching](activation-patching.md) — pair logit lens with patching to
  localize where a fact lives.
- [attention-patterns](attention-patterns.md)
- [probing](probing.md) — the other read-only depth measurement, with controls.
- `docs/usage/access-and-modify.md`
- https://nnsight.net/notebooks/tutorials/logit_lens/
- nostalgebraist (2020), "interpreting GPT: the logit lens".

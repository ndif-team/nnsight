---
title: Logit Lens
one_liner: Apply the final layer norm and unembedding to intermediate hidden states to read off what each layer "thinks" the next token is.
tags: [pattern, interpretability, residual-stream, decoding]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/patterns/activation-patching.md, docs/patterns/probing.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/modeling/transformers.py]
---

# Logit Lens

## What this is for

The logit lens (nostalgebraist, 2020) reads the residual stream at every layer
through the model's own final norm and unembedding: `lm_head(ln_f(h_L))` for each
layer L. It asks "if the model stopped here, what would it predict?" — use it to see
where in depth an answer emerges, or to check that a model knows a fact before
running invasive interventions.

Calling `model.lm_head(...)` inside a trace runs the module with the trace **stood
down** — just the linear math, applied out of order without re-triggering the model.
Reading `model.lm_head.output` instead would intercept the model's own `lm_head`
call, a different operation. See
[access-and-modify](../usage/access-and-modify.md) ("Calling modules directly inside
a trace").

On `VLLM` the lens goes through the model's logits path instead — see
[vllm-parallelism.md](../models/vllm-parallelism.md#tensor-parallelism-is-transparent).

## Canonical pattern

One trace decodes every layer, tracks one target token, and checks the wiring:

```python
import torch
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer.encode(" Paris")[0]

with model.trace(prompt):
    top, p_paris = nnsight.save([]), nnsight.save([])
    for block in model.transformer.h:
        hs = block.output                    # residual stream: a plain [B, S, hidden] tensor
        probs = model.lm_head(model.transformer.ln_f(hs))[0, -1].softmax(-1)
        top.append(probs.argmax().item())
        p_paris.append(probs[paris].item())
    # wiring check: the lens on the last block must be the model's own logits
    lens = model.lm_head(model.transformer.ln_f(model.transformer.h[-1].output)).save()
    real = model.output.logits.save()

assert torch.equal(lens, real), (lens - real).abs().max()
for i, (t, p) in enumerate(zip(top, p_paris)):
    print(f"layer {i:2d}: {model.tokenizer.decode(t)!r:<10} P(' Paris') = {p:.4f}")
```

```
layer  0: ' the'     P(' Paris') = 0.0000
layer  1: ' the'     P(' Paris') = 0.0000
layer  2: ' the'     P(' Paris') = 0.0000
layer  3: ' the'     P(' Paris') = 0.0000
layer  4: ' the'     P(' Paris') = 0.0001
layer  5: ' the'     P(' Paris') = 0.0004
layer  6: ' East'    P(' Paris') = 0.0014
layer  7: ' Ing'     P(' Paris') = 0.0028
layer  8: ' Rome'    P(' Paris') = 0.0255
layer  9: ' London'  P(' Paris') = 0.2484
layer 10: ' Paris'   P(' Paris') = 0.1834
layer 11: ' Paris'   P(' Paris') = 0.0700
```

`P(' Paris')` peaks at layer 9 (`0.248`), where `' London'` is still top-1; `' Paris'`
only becomes top-1 at layer 10, and the final layer is *less* confident (`0.070`)
than either. Argmax alone shows none of that.

A block's `.output` is the residual tensor itself — `block.output[0]` indexes the
batch, not a tuple. Re-reading `h[-1].output` after the loop has already passed it is
allowed: forward order constrains only the *first* read of a module in an invoke (see
[out-of-order-error](../errors/out-of-order-error.md)).

## Check the wiring

Applied to the *last* block, the lens is the model's own final computation, so it has
to reproduce the model's logits exactly. The `torch.equal` line above says whether the
norm, the head and the block output you picked are the right three — keep it in
before you plot anything.

`True` on GPT-2, SmolLM2-135M, Qwen2.5-0.5B and pythia-70m. A `False` means one of
three things, all of which produce a plot rather than an error if you skip this:

- the wrong norm, or none. Without `ln_f`, GPT-2 decodes `' the'` at *every* layer
  with probability `1.0000` — a lens that looks more confident than the correct one;
- the wrong head (`model.embed_out` does not exist on pythia; it is `model.lm_head`);
- or the model post-processes the head's output.

The third case is Gemma-2, which applies `tanh` logit softcapping in
`Gemma2ForCausalLM.forward`, after `lm_head`. On `google/gemma-2-2b` the check fails
by `51.0`, and the uncapped distribution is far too confident: entropy `0.0050`
against the model's own `0.6010`. The top-1 token usually survives; nothing else
does. Apply the same cap yourself:

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

### Tuned lens

A tuned lens ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112)) learns one
affine translator per layer, applied *before* the frozen `ln_f` and `lm_head`:
`model.lm_head(model.transformer.ln_f(tuned_maps[L](block.output)))`. Keep `ln_f`:
a LayerNorm is not affine, so dropping it and decoding `lm_head(A_L(hs))` is a
different model, not a reparameterization.

### Attention-only / MLP-only lens

Read the sub-block outputs instead of the block residual: `block.mlp.output` (a
tensor) and `block.attn.output[0]` (the first element of the attention tuple).

## Interpretation tips

- **Look at the layer where the answer first becomes top-1.** That layer does the
  bulk of the decision; later layers usually refine.
- **GPT-2's smooth trajectory is not the general case.** Qwen2.5-0.5B and
  SmolLM2-135M decode punctuation or `' the'` until their last few layers, then
  answer correctly. Once the wiring check passes, a flat curve is as often a fact
  about the lens as about the model.
- **Position matters.** `[:, -1, :]` reads the next-token prediction. For factual
  recall the relevant position is often the subject token.
- **Module names differ.** GPT-2 uses `model.transformer.h[i]` and
  `model.transformer.ln_f`; Llama / Mistral / Qwen typically use
  `model.model.layers[i]` and `model.model.norm`. Use `print(model)`.

## Related

- [activation-patching](activation-patching.md) — pair logit lens with patching to
  localize where a fact lives.
- [probing](probing.md) — the other read-only depth measurement, with controls.
- [access-and-modify](../usage/access-and-modify.md)
- https://nnsight.net/notebooks/tutorials/logit_lens/
- nostalgebraist (2020), "interpreting GPT: the logit lens".

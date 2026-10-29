---
title: Patterns Index
one_liner: Cookbook of interpretability recipes built on nnsight - logit lens, patching, ablation, steering, attribution, and more.
tags: [pattern, interpretability, index]
related: [docs/usage/index.md, docs/concepts/index.md, docs/errors/index.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Patterns Index

Up: [docs/index.md](../index.md) — all sections.

One technique per page. Examples use `openai-community/gpt2` so they run on a
laptop. API reference: [usage/index.md](../usage/index.md); something breaking:
[errors/symptom-index.md](../errors/symptom-index.md).

## Read what the model computes

- [logit-lens](logit-lens.md) — Decode each layer's residual through the final norm and unembedding to see where in depth an answer emerges.
- [attention-patterns](attention-patterns.md) — Read attention probabilities (`attn.output[1]`, or `.source` for the raw softmax) to see what each head attends to.
- [probing](probing.md) — Train a linear classifier on activations to test what is decodable, where in depth it appears, and whether the model uses it.
- [sae-and-auxiliary-modules](sae-and-auxiliary-modules.md) — Run an SAE, transcoder, adapter or probe on a layer: read its features, splice in its reconstruction, or make a parameter-only SAE observable with `WrapperModule`.

## Test what is causal

- [activation-patching](activation-patching.md) — Copy activations from a clean run into a corrupt one to localize where a fact or behavior lives (IOI-style, layer × position maps).
- [attribution-patching](attribution-patching.md) — Approximate a whole patching map from one clean forward and one corrupt forward+backward.
- [ablation](ablation.md) — Zero, mean or resample a layer, head, neuron or MoE expert and measure what breaks.
- [per-head-attention](per-head-attention.md) — Read, ablate or patch a single attention head.
- [gradient-based-attribution](gradient-based-attribution.md) — Saliency, input × gradient and integrated gradients on any activation.

## Change behavior

- [steering](steering.md) — Add a direction to the residual stream to push behavior, including during generation, and choose the coefficient.

## Run at scale on NDIF

- [remote-dataset-sweep](remote-dataset-sweep.md) — Run a model over a dataset remotely: load the data on the server, loop in one session, bring back only the reduction.
- [remote-training](remote-training.md) — Train a LoRA, steering vector or probe remotely, with the whole optimizer loop server-side.

Notebook versions of these recipes: [nnsight.net tutorials](https://nnsight.net/tutorials.html).

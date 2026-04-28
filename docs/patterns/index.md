---
title: Patterns Index
one_liner: Cookbook of interpretability recipes built on nnsight - logit lens, patching, ablation, steering, attribution, and more.
tags: [pattern, interpretability, index]
related: [docs/usage/index.md, docs/concepts/index.md, docs/gotchas/index.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracing/tracer.py]
---

# Patterns Index

Recipe-style pages, one technique per file. Each page opens with the smallest working example, then variations, then interpretation tips and gotchas. All examples use a small open-weights `LanguageModel` (typically `openai-community/gpt2`) so they run on a laptop.

If you are looking for the underlying API (e.g. `tracer.invoke`, `tensor.backward`, `eproperty`), see `docs/usage/index.md`. If something is breaking, start at `docs/gotchas/`.

## Look at activations

What is happening inside the model on a given prompt?

- [logit-lens](logit-lens.md) - Apply the final norm + unembedding to every layer's residual to "decode" what each layer is thinking.
- [attention-patterns](attention-patterns.md) - Extract the attention probability matrix from a transformer block via `.source`.
- [sae-and-auxiliary-modules](sae-and-auxiliary-modules.md) - Wire an SAE (or any auxiliary module) into a model and trace through it as a first-class submodule.

## Modify activations

Change the model's internal state and observe the effect on output.

- [activation-patching](activation-patching.md) - Replace activations from one run into another. Causal mediation analysis, IOI-style patching.
- [ablation](ablation.md) - Zero / mean / noise ablate specific components, positions, or features and measure the change.
- [steering](steering.md) - Add a precomputed direction to the residual stream to push behavior in a target direction.

## Compare runs

Multiple prompts, multiple invokes, attribution in one batch.

- [multi-prompt-comparison](multi-prompt-comparison.md) - Multiple `tracer.invoke(...)` calls in one trace, empty invokes for batch-wide ops, and when you need a barrier.
- [attribution-patching](attribution-patching.md) - Linear approximation of activation patching using corrupt-run gradients times clean-vs-corrupt activation differences.

## Gradients

Backprop-based interpretability.

- [gradient-based-attribution](gradient-based-attribution.md) - `with logits.sum().backward():` to compute saliency, integrated gradients, and per-component attribution.

## Heads

Per-attention-head access.

- [per-head-attention](per-head-attention.md) - Two ways to slice attention output into heads: in-trace reshape, and a custom `Envoy` with `eproperty.transform`.

## Other resources

- [nnsight.net tutorials](https://nnsight.net/tutorials.html) - Mirror of these patterns with notebooks.
- `docs/usage/index.md` - Reference for every API surface used in these recipes.
- `NNsight.md` - Internal architecture (tracing, threading, hooks).

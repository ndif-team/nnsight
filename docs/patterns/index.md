---
title: Patterns Index
one_liner: Cookbook of interpretability recipes built on nnsight - logit lens, patching, ablation, steering, attribution, and more.
tags: [pattern, interpretability, index]
related: [docs/usage/index.md, docs/concepts/index.md, docs/errors/index.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Patterns Index

Recipe-style pages, one technique per file. Each opens with the smallest working
example, then variations, then interpretation tips and gotchas. All examples use a
small open-weights model — `TransformersModel("openai-community/gpt2", dispatch=True)`
— so they run on a laptop.

For the underlying API (`tracer.invoke`, `tensor.backward`, `edit`, `skip`,
`source`), see `docs/usage/index.md`. If something is breaking, start at
`docs/errors/`.

## Look at activations

What is happening inside the model on a given prompt?

- [logit-lens](logit-lens.md) — Apply the final norm + unembedding to every layer's residual to "decode" what each layer is thinking.
- [attention-patterns](attention-patterns.md) — Extract the attention probability matrix from a block via `.source`.
- [sae-and-auxiliary-modules](sae-and-auxiliary-modules.md) — Attach an SAE (or any auxiliary module) via assignment + `edit()`, route through it with `hook=True`, and read its internals.

## Modify activations

Change the model's internal state and observe the effect on output.

- [activation-patching](activation-patching.md) — Replace activations from one run into another. Causal mediation, IOI-style patching.
- [ablation](ablation.md) — Zero / mean / noise ablate specific components, positions, or features and measure the change.
- [steering](steering.md) — Add a precomputed direction to the residual stream to push behavior in a target direction.

## Compare runs

Multiple prompts, multiple invokes, attribution in one batch.

- [multi-prompt-comparison](multi-prompt-comparison.md) — Multiple `tracer.invoke(...)` in one trace, empty invokes for batch-wide ops, and `tracer.barrier(n)` for cross-invoke sharing.
- [attribution-patching](attribution-patching.md) — Linear approximation of activation patching from corrupt-run gradients times clean-vs-corrupt activation differences.

## Gradients

Backprop-based interpretability.

- [gradient-based-attribution](gradient-based-attribution.md) — `with metric.backward():` to read `.grad` for saliency, integrated gradients, and per-component attribution.

## Heads

Per-attention-head access.

- [per-head-attention](per-head-attention.md) — Slice attention output into heads for per-head reading and editing.

## Working at scale on NDIF

One session per experiment, and as few bytes as possible in each direction.

- [remote-dataset-sweep](remote-dataset-sweep.md) — Load the data on the server rather than shipping it, loop inside one session, bring back the reduction.
- [remote-training](remote-training.md) — Fit a LoRA / steering vector / probe with the whole optimizer loop server-side.

## Other resources

- [nnsight.net tutorials](https://nnsight.net/tutorials.html) — Notebook mirror of these patterns.
- `docs/usage/index.md` — Reference for every API surface used in these recipes.
- `docs/concepts/index.md` — Internal architecture (interleaver, envoy, greenlets/mediators).

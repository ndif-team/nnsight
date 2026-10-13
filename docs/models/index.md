---
title: Model Classes
one_liner: Decision tree for picking the right nnsight model wrapper.
tags: [models, index]
related: [docs/index.md, docs/models/transformers-model.md, docs/models/tensor-parallel.md, docs/models/quantization.md, docs/models/nnsight-base.md, docs/models/diffusion-model.md, docs/models/vllm.md, docs/models/language-model.md, docs/models/vision-language-model.md]
sources: [src/nnsight/__init__.py, src/nnsight/modeling/base.py, src/nnsight/modeling/transformers.py, src/nnsight/modeling/diffusion.py, src/nnsight/modeling/vllm/vllm.py, src/nnsight/modeling/language.py, src/nnsight/modeling/vlm.py]
---

# Model Classes

Up: [docs/index.md](../index.md) — all sections.

## What this is for

Pick the model wrapper that matches what you have. All wrappers expose the same tracing API (`.trace()`, `.generate()`, `.scan()`, `.edit()`, `.session()`, `.cache()`) but differ in how they load the model, prepare inputs, and batch across invokes.

## Decision tree

- You have a **HuggingFace model** (any task — LMs, VLMs, classifiers, encoders) and want tokenization/featurization, batching, and generation for free.
  - See [docs/models/transformers-model.md](transformers-model.md). Class: `nnsight.TransformersModel`. **This is the primary HF wrapper — start here.**

- You have an **arbitrary `torch.nn.Module`** (custom net, research code, non-HF model) and just want intervention access.
  - See [docs/models/nnsight-base.md](nnsight-base.md). Class: `nnsight.NNsight`.

- You have a **diffusers `DiffusionPipeline`** (Stable Diffusion, Flux, DiT, SDXL, etc.) and want to trace UNet / transformer / VAE / text-encoder activations.
  - See [docs/models/diffusion-model.md](diffusion-model.md). Class: `nnsight.DiffusionModel`.

- You need **production throughput, continuous batching, or async streaming** with NNsight interventions.
  - See [docs/models/vllm.md](vllm.md). Class: `nnsight.modeling.vllm.VLLM`.
  - An edit every request on the engine runs, including other clients': [vllm-editing.md](vllm-editing.md).
  - Streaming with `mode="async"`, or one engine behind HTTP (`nnsight-serve`): [vllm-serving.md](vllm-serving.md).
  - Tensor parallelism, MoE and hybrid models on vLLM, and version notes: [vllm-parallelism.md](vllm-parallelism.md).

- Your model is **too big for one GPU** and you want to trace it split across several with `transformers` tensor parallelism (one process per GPU, launched with `torchrun`).
  - See [docs/models/tensor-parallel.md](tensor-parallel.md). Still `TransformersModel` — pass `distributed_config=DistributedConfig(tp_size=N)`; sharded activations are gathered for you.

- Your model is **too big for one GPU** and you would rather narrow it than split it.
  - See [docs/models/quantization.md](quantization.md). Still `TransformersModel` — pass `dtype="nf4"` (or `int8`, `fp8`, ...) where you would pass a torch dtype. Module paths and activations are unchanged.

### Deprecated aliases

- `nnsight.LanguageModel` — DEPRECATED thin alias for `TransformersModel(repo_id, task="text-generation")`. Warns on construction. See [docs/models/language-model.md](language-model.md).
- `nnsight.VisionLanguageModel` — DEPRECATED thin alias for `TransformersModel(repo_id, task="image-text-to-text")`. Warns on construction. See [docs/models/vision-language-model.md](vision-language-model.md).

## Related

- [docs/concepts/](../concepts/index.md) — the underlying tracing / Envoy / interleaver model
- [docs/remote/](../remote/index.md) — running these on NDIF
- [docs/errors/symptom-index.md](../errors/symptom-index.md) — something broke

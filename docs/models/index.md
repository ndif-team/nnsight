---
title: Model Classes
one_liner: Decision tree for picking the right nnsight model wrapper.
tags: [models, index]
related: [docs/models/transformers-model.md, docs/models/tensor-parallel.md, docs/models/nnsight-base.md, docs/models/diffusion-model.md, docs/models/vllm.md, docs/models/language-model.md, docs/models/vision-language-model.md]
sources: [src/nnsight/__init__.py:53, src/nnsight/modeling/base.py:6, src/nnsight/modeling/transformers.py:161, src/nnsight/modeling/diffusion.py:38, src/nnsight/modeling/vllm/vllm.py:36, src/nnsight/modeling/language.py:19, src/nnsight/modeling/vlm.py:29]
---

# Model Classes

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

- Your model is **too big for one GPU** and you want to trace it split across several with `transformers` tensor parallelism (one process per GPU, launched with `torchrun`).
  - See [docs/models/tensor-parallel.md](tensor-parallel.md). Still `TransformersModel` — pass `distributed_config=DistributedConfig(tp_size=N)`; sharded activations are gathered for you.

### Deprecated aliases

- `nnsight.LanguageModel` — DEPRECATED thin alias for `TransformersModel(repo_id, task="text-generation")`. Warns on construction. See [docs/models/language-model.md](language-model.md).
- `nnsight.VisionLanguageModel` — DEPRECATED thin alias for `TransformersModel(repo_id, task="image-text-to-text")`. Warns on construction. See [docs/models/vision-language-model.md](vision-language-model.md).

## At a glance

| Class | Import | Use it for | Loader backend |
|-------|--------|------------|----------------|
| `TransformersModel` | `nnsight` | **Any** HF transformers task | `transformers.pipeline` |
| `NNsight` | `nnsight` | Any `torch.nn.Module` | None (you instantiate) |
| `DiffusionModel` | `nnsight` | diffusers pipelines | `diffusers.DiffusionPipeline` |
| `VLLM` | `nnsight.modeling.vllm` | High-throughput serving with interventions | `vllm.LLM` (sync) / `vllm.v1.engine.async_llm.AsyncLLM` (async) |
| `TransformersModel` + `DistributedConfig` | `nnsight` + `transformers.distributed` | A model sharded across GPUs (tensor parallel) | `transformers.pipeline` |
| `LanguageModel` *(deprecated)* | `nnsight` | → `TransformersModel(task="text-generation")` | `transformers.pipeline` |
| `VisionLanguageModel` *(deprecated)* | `nnsight` | → `TransformersModel(task="image-text-to-text")` | `transformers.pipeline` |

All HF-backed wrappers support lazy loading: `dispatch=False` (default) downloads only configs and builds a meta-tensor architecture; `dispatch=True` loads real weights immediately. Real weights also load automatically on the first `.trace()` / `.generate()` / `.pipe()` call. `.scan()` never dispatches.

## Related

- [docs/concepts/](../concepts/) — the underlying tracing / Envoy / interleaver model
- [docs/remote/](../remote/) — running these on NDIF
- [docs/gotchas/](../gotchas/) — cross-cutting pitfalls

---
title: Model Classes
one_liner: Decision tree for picking the right nnsight model wrapper.
tags: [models, index]
related: [docs/models/nnsight-base.md, docs/models/language-model.md, docs/models/vision-language-model.md, docs/models/diffusion-model.md, docs/models/vllm.md]
sources: [src/nnsight/__init__.py:69, src/nnsight/modeling/base.py:8, src/nnsight/modeling/language.py:21, src/nnsight/modeling/vlm.py:17, src/nnsight/modeling/diffusion.py:221, src/nnsight/modeling/vllm/vllm.py:43]
---

# Model Classes

## What this is for

Pick the model wrapper that matches the runtime and architecture you have. All wrappers expose the same tracing API (`.trace()`, `.generate()`, `.scan()`, `.edit()`, `.session()`) but differ in how they load the model, prepare inputs, and batch across invokes.

## Decision tree

- You have an **arbitrary `torch.nn.Module`** (custom net, research code, non-HF model) and just want intervention access.
  - See [docs/models/nnsight-base.md](nnsight-base.md). Class: `nnsight.NNsight`.

- You have a **HuggingFace causal language model** (GPT-2, Llama, Mistral, Qwen, etc.) and want tokenization, batching, and `.generate()` for free.
  - See [docs/models/language-model.md](language-model.md). Class: `nnsight.LanguageModel`.

- You have a **vision-language model** (LLaVA, Qwen2-VL, InternVL, Phi-3.5-vision, etc.) that uses an `AutoProcessor` for text+images.
  - See [docs/models/vision-language-model.md](vision-language-model.md). Class: `nnsight.VisionLanguageModel`.

- You have a **diffusers `DiffusionPipeline`** (Stable Diffusion, Flux, DiT, SDXL, etc.) and want to trace UNet / transformer / VAE / text-encoder activations across denoising steps.
  - See [docs/models/diffusion-model.md](diffusion-model.md). Class: `nnsight.DiffusionModel`.

- You need **production throughput, tensor parallelism, continuous batching, or async streaming** with NNsight interventions.
  - See [docs/models/vllm.md](vllm.md). Class: `nnsight.modeling.vllm.VLLM`.

## At a glance

| Class | Module | Use it for | Loader backend |
|-------|--------|------------|----------------|
| `NNsight` | `nnsight` | Any `torch.nn.Module` | None (you instantiate) |
| `LanguageModel` | `nnsight` | HF causal LMs | `transformers.AutoModelForCausalLM` |
| `VisionLanguageModel` | `nnsight` | HF VLMs (text+images) | `transformers.AutoModelForImageTextToText` + `AutoProcessor` |
| `DiffusionModel` | `nnsight` | diffusers pipelines | `diffusers.DiffusionPipeline` |
| `VLLM` | `nnsight.modeling.vllm` | High-throughput serving with interventions | `vllm.LLM` (sync) / `vllm.v1.engine.async_llm.AsyncLLM` (async) |

All HF-backed wrappers (`LanguageModel`, `VisionLanguageModel`, `DiffusionModel`, `VLLM`) inherit `MetaMixin` and support lazy loading: pass `dispatch=False` (default) to download only configs and build a meta-tensor architecture, or `dispatch=True` to load real weights immediately. See [docs/models/index.md](../models/index.md) if it exists for details. Real weights load automatically on the first `.trace()` / `.generate()` call.

## Related

- [docs/concepts/](../concepts/) for the underlying tracing/Envoy/interleaver model
- [docs/remote/](../remote/) for running any of these on NDIF
- [docs/gotchas/](../gotchas/) for cross-cutting pitfalls

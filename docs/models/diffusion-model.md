---
title: DiffusionModel
one_liner: Wraps any diffusers DiffusionPipeline (UNet- or transformer-based) with NNsight tracing.
tags: [models, diffusion, diffusers]
related: [docs/models/index.md, docs/models/language-model.md]
sources: [src/nnsight/modeling/diffusion.py:221, src/nnsight/intervention/batching.py, tests/test_diffusion.py]
---

# DiffusionModel

## What this is for

`nnsight.DiffusionModel` wraps any `diffusers.DiffusionPipeline` so you can trace and intervene on its sub-modules — UNet (Stable Diffusion), transformer (Flux, DiT), VAE, text encoder — with the same NNsight API as language models. It supports:

- Single-step tracing (`.trace()` defaults to `num_inference_steps=1` for fast exploration)
- Full-pipeline generation (`.generate()` runs all denoising steps)
- Iterating across denoising steps with `tracer.iter[:]`
- Batching multiple prompts via invokes
- Lazy meta-tensor loading — only config files are downloaded with `dispatch=False`
- Works with both UNet-based pipelines (Stable Diffusion 1.x/2.x/XL) and transformer-based pipelines (Flux, DiT, SD3)

This is a less-common path than `LanguageModel` but follows the same patterns.

## When to use / when not to use

Use `DiffusionModel` when:
- You have a diffusers pipeline (`StableDiffusionPipeline`, `FluxPipeline`, `DiTPipeline`, etc.) loadable from a HuggingFace repo.
- You want to study denoising trajectories, intervene on the U-Net / transformer / VAE / text encoder, or capture activations across inference steps.
- You're researching mechanistic interpretability of diffusion models, prompt steering, or representation engineering on image generation.

Do not use `DiffusionModel` when:
- Your pipeline isn't a `diffusers.DiffusionPipeline` subclass — wrap the underlying `torch.nn.Module` directly with [`NNsight`](nnsight-base.md).
- You need vLLM-style serving — vLLM is for LLMs, not diffusion pipelines.

## Loading

```python
from nnsight import DiffusionModel
import torch

sd = DiffusionModel(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    safety_checker=None,
    dispatch=True,
)
```

### Constructor

```python
DiffusionModel(
    repo_id,
    *,
    automodel=DiffusionPipeline,    # diffusers pipeline class or string name
    revision=None,
    dispatch=False,
    meta_buffers=True,
    rename=None,
    envoys=None,
    import_edits=False,
    **kwargs,                       # forwarded to the pipeline's from_pretrained()
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | HuggingFace repo ID (e.g. `"stabilityai/stable-diffusion-2-1"`, `"black-forest-labs/FLUX.1-schnell"`). |
| `automodel` | The diffusers pipeline class. Defaults to `DiffusionPipeline` (which auto-resolves the right subclass from `model_index.json`). Can also be a string name resolvable from `diffusers.pipelines` (e.g. `"FluxPipeline"`). See `diffusion.py:266`. |
| `dispatch` | If `True`, real weights download immediately via `Diffuser(automodel, repo_id, ...)`. If `False`, only `model_index.json` and per-component `config.json` files are downloaded — every `nn.Module` component is built on the meta device via `_build_pipeline_from_config` (`diffusion.py:58`). |
| `device_map` | Defaults to `"balanced"` if `None` or `"auto"` is passed. See `diffusion.py:342`. |
| `safety_checker=None`, `feature_extractor=None`, etc. | Pipeline component overrides. Forwarded to the pipeline constructor. |
| `torch_dtype` | Forwarded to `from_pretrained` for real loading. Note: it is filtered out of meta-loading (`diffusion.py:152`) because pipeline `__init__` doesn't accept it. |

All other `**kwargs` go to the pipeline's `from_pretrained()`.

### Dispatch behavior

- `dispatch=False` (default) downloads only the configs (a few KB total) and instantiates each `nn.Module` component on the `meta` device via `cls.from_config(...)` or `cls(auto_cfg)`. Tokenizers are loaded normally (lightweight). Schedulers and feature extractors are set to `None` and only loaded on real dispatch. See `_build_pipeline_from_config` at `diffusion.py:58`.
- The pipeline's full architecture is wrapped in a `Diffuser` (`diffusion.py:164`) that exposes every `nn.Module` component as a sub-attribute, so you can write `sd.unet.output`, `sd.text_encoder.output`, etc., before any weights are loaded.
- First `.trace()` / `.generate()` call triggers `.dispatch()` automatically.

## Canonical pattern

```python
from nnsight import DiffusionModel
import torch

sd = DiffusionModel(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    safety_checker=None,
    dispatch=True,
)

# Single-step trace (fast)
with sd.trace("A photo of a cat") as tracer:
    denoiser_out = sd.unet.output.save()
    output = tracer.result.save()

print(denoiser_out[0].shape)
output.images[0].save("cat.png")
```

`.trace()` defaults to `num_inference_steps=1` (set in `__call__` at `diffusion.py:471`). Override by passing `num_inference_steps=N` explicitly.

### Full generation across steps

```python
with sd.generate("A cat", num_inference_steps=50, seed=42) as tracer:
    denoiser_outputs = list().save()
    for step in tracer.iter[:]:
        denoiser_outputs.append(sd.unet.output[0].clone())
    output = tracer.result.save()

assert len(denoiser_outputs) == 50
output.images[0].save("cat_50steps.png")
```

`.generate()` does **not** override `num_inference_steps`, so the pipeline's own default (or your explicit value) takes effect.

### Transformer-based pipelines (Flux, DiT, SD3)

Same API, different denoiser attribute name:

```python
flux = DiffusionModel("black-forest-labs/FLUX.1-schnell", dispatch=True)

with flux.trace("A cat"):
    transformer_out = flux.transformer.output.save()
```

### Intervening on the denoiser

```python
# Zero out the UNet output — drastically changes the image
with sd.trace("A cat", num_inference_steps=1) as tracer:
    sd.unet.output[0][:] = 0
    output = tracer.result.save()
```

### Batching multiple prompts

`DiffusionBatcher` (in `nnsight/intervention/batching.py`) automatically handles three batch-size scenarios:

1. **Plain prompts** — one row per prompt
2. **`num_images_per_prompt`** — N rows per prompt
3. **Classifier-free guidance** (`guidance_scale > 1`) — doubles the batch (uncond + cond)

```python
with sd.generate(num_inference_steps=20, num_images_per_prompt=3, seed=423) as tracer:
    with tracer.invoke("a cat"):
        out_cat = sd.unet.output[0].save()                          # 3 rows
    with tracer.invoke(["a panda", "a birthday cake"]):
        out_pair = sd.unet.output[0].save()                         # 6 rows
    with tracer.invoke("a wave"):
        out_wave = sd.unet.output[0].save()                         # 3 rows
    with tracer.invoke():
        out_all = sd.unet.output[0].save()                          # 12 rows
```

With `guidance_scale=7.5`, those become 6 / 12 / 6 / 24 rows respectively (CFG doubles each). See `tests/test_diffusion.py:184` for the full set of batching tests.

### Seed reproducibility

Pass `seed=N` to either `.trace()` or `.generate()`:

```python
with sd.generate("A cat", num_inference_steps=2, seed=42) as tracer:
    output1 = tracer.result.save()

with sd.generate("A cat", num_inference_steps=2, seed=42) as tracer:
    output2 = tracer.result.save()

# output1.images[0] == output2.images[0] (pixel-exact)
```

When multiple prompts are batched, each gets `seed + offset` to avoid identical noise (`diffusion.py:435`).

### Accessing the underlying `DiffusionPipeline`

For non-traced operations on the raw diffusers pipeline (saving, scheduler swaps, `pipeline.to(...)`, attention slicing, custom call paths, etc.), reach through `model._model` — that's the `Diffuser` wrapper — and grab its `.pipeline` attribute:

```python
pipeline = model._model.pipeline       # the raw diffusers DiffusionPipeline

# Anything diffusers supports works directly on it
pipeline.scheduler = SomeOtherScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_attention_slicing()
pipeline.save_pretrained("./my-finetune")

# You can also call it directly (bypasses NNsight entirely — same as the snippet below)
output = pipeline("A cat", num_inference_steps=20)
```

Source: `src/nnsight/modeling/diffusion.py:193` (`Diffuser.__init__` stores `self.pipeline`) and `:218` (`Diffuser.forward` delegates `__call__` to `self.pipeline`).

Once you've mutated the pipeline in place, the next `model.trace()` / `model.generate()` call uses the modified pipeline. NNsight only re-wraps the `nn.Module` components on construction, so swapping non-module components (like the scheduler) at any time is fine.

### Run without a tracing context

```python
output = sd.generate("A cat", num_inference_steps=20)
output.images[0].save("cat.png")
```

This bypasses NNsight entirely — no interleaver, no envoys, just the underlying pipeline call.

## Special properties

| Attribute | Description | Source |
|-----------|-------------|--------|
| `model.unet` / `model.transformer` | The denoiser. Attribute name depends on the pipeline (UNet-based vs transformer-based). | `Diffuser.__init__` at `diffusion.py:201` |
| `model.vae` | The VAE encoder/decoder. | same |
| `model.text_encoder` (and `model.text_encoder_2` for SDXL/Flux) | Text encoder(s). Only `nn.Module` components are wrapped as Envoys. | same |
| `model.tokenizer` | Pipeline's tokenizer. Loaded eagerly even in meta mode. | `_build_pipeline_from_config` at `diffusion.py:120` |
| `model.config` | Pipeline `model_index.json` config dict. | `diffusion.py:285` |
| `model.automodel` | The diffusers pipeline class used for loading. | `diffusion.py:270` |
| `model._model` | The underlying `Diffuser` wrapper. Use `model._model.pipeline` to reach the raw diffusers `DiffusionPipeline` (for non-traced operations — see [Accessing the underlying DiffusionPipeline](#accessing-the-underlying-diffusionpipeline)). | `diffusion.py:277`, `diffusion.py:193` |
| `tracer.result` (inside trace) | The final pipeline output (typically a dataclass with `.images`). |

The pipeline's **scheduler** is **not** wrapped as an Envoy — only `nn.Module` and `PreTrainedTokenizerBase` components are (`diffusion.py:201`). To intervene on scheduler behavior you'd modify the pipeline directly.

## Limitations

- **Scheduler / non-module components are not Envoy-wrapped.** Only `torch.nn.Module` and `PreTrainedTokenizerBase` instances are exposed as sub-envoys (`diffusion.py:201`). If you need to intervene on a scheduler, do it pre-trace.
- **`torch_dtype` only applies during real dispatch.** Meta-loading filters it out because pipeline constructors don't accept it (`diffusion.py:152`).
- **Custom pipelines outside `diffusers.pipelines`** may fail to resolve in `_resolve_component_cls` (`diffusion.py:19`). The function tries `diffusers`, `transformers`, and `diffusers.pipelines.<lib_name>`. Components it can't resolve are set to `None` in meta mode and only built on real dispatch.
- **`.trace()` defaults to 1 step.** If you forget to pass `num_inference_steps=`, you get a 1-step image which is essentially noise. This is intentional for fast tracing — call `.generate()` for normal output.
- **No remote execution yet.** `DiffusionModel` is a `HuggingFaceModel` subclass and inherits remote infrastructure, but as of this writing diffusion models are not deployed on NDIF (see 0.6.0 release notes).

## Gotchas

- **Tuple vs tensor outputs.** UNet returns a tuple in some diffusers versions and a single tensor in others. Use `sd.unet.output[0]` defensively, or `if isinstance(out, tuple)` checks.
- **CFG doubles your batch silently.** `guidance_scale > 1` (the default for most SD pipelines) means every batch row appears twice. Account for this when slicing batched outputs (see the batching tests in `tests/test_diffusion.py:220`).
- **`device_map="auto"` is rewritten to `"balanced"`** for diffusion pipelines because diffusers expects that string (`diffusion.py:342`).
- **Meta loading downloads small config files.** `dispatch=False` is not a hard offline mode — it still hits the HF Hub for `model_index.json` and each component's `config.json`. This is by design, so the Envoy tree mirrors the real architecture.

## Related

- [docs/models/index.md](index.md) — pick the right wrapper
- [docs/models/nnsight-base.md](nnsight-base.md) — base class
- `src/nnsight/modeling/diffusion.py` — source (Diffuser wrapper, DiffusionModel class, `_build_pipeline_from_config`)
- `src/nnsight/intervention/batching.py` — `DiffusionBatcher` implementation
- `tests/test_diffusion.py` — runnable examples for tracing, generation, batching, CFG, swapping, seeded reproducibility, meta loading, and Flux

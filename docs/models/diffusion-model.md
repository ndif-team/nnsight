---
title: DiffusionModel
one_liner: Wraps any diffusers DiffusionPipeline (UNet- or transformer-based) with NNsight tracing.
tags: [models, diffusion, diffusers]
related: [docs/models/index.md, docs/models/nnsight-base.md, docs/models/transformers-model.md]
sources: [src/nnsight/modeling/diffusion.py, src/nnsight/modeling/huggingface.py, tests/test_diffusion.py]
---

# DiffusionModel

## What this is for

`nnsight.DiffusionModel` wraps any `diffusers.DiffusionPipeline` — UNet-based (SD 1.x/2.x/XL) or transformer-based (Flux, DiT, SD3) — so you can trace and intervene on its `nn.Module` components (denoiser, VAE, text encoders) with the same API as other models, across denoising steps and per prompt. A pipeline that isn't a `DiffusionPipeline` goes through [`NNsight`](nnsight-base.md) on its underlying module.

## Loading

```python
from nnsight import DiffusionModel

sd = DiffusionModel("stabilityai/stable-diffusion-2-1", dispatch=True)
```

### Constructor

```python
DiffusionModel(
    repo_id,
    *,
    automodel=None,         # diffusers class to load through (class or name)
    revision=None,          # git branch / tag / commit
    dispatch=False,         # True = load real weights now; False = lazy meta build
    rename=None,            # dict of module-path aliases
    **kwargs,               # forwarded to DiffusionPipeline.from_pretrained()
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | HuggingFace repo id (e.g. `"stabilityai/stable-diffusion-2-1"`, `"black-forest-labs/FLUX.1-schnell"`). |
| `automodel` | The diffusers class to load through, as a class or a name in `diffusers`. Default `None` means `DiffusionPipeline`, which builds whatever class the repo's `model_index.json` declares. See [A different task from the same weights](#a-different-task-from-the-same-weights). |
| `dispatch` | `True` loads real weights via `DiffusionPipeline.from_pretrained` during `__init__`. `False` (default) builds a meta pipeline: each `nn.Module` component from its config on the `meta` device, light components (schedulers, tokenizers) loaded normally (`_build_component`). |
| `rename` | Module-path aliases, e.g. `{"unet": "denoiser"}`. |
| `torch_dtype`, `safety_checker=None`, `variant`, ... | Forwarded to `from_pretrained` for real loading. |

### A different task from the same weights

Every Stable Diffusion repo declares the text-to-image pipeline, so `image=` has nowhere to go even though the weights serve image-to-image. Name the class to load through with `automodel=`:

```python
from diffusers import AutoPipelineForImage2Image
from nnsight import DiffusionModel

model = DiffusionModel("stable-diffusion-v1-5/stable-diffusion-v1-5",
                       automodel=AutoPipelineForImage2Image)

with model.trace("a photo of a cat", image=source, strength=0.5,
                 num_inference_steps=20):
    latents = model.unet.output[0].save()
    result = model.output.save()
```

The `AutoPipelineFor*` classes (`Text2Image`, `Image2Image`, `Inpainting`, `Text2Audio`) pick the right class per architecture; a concrete class or its name works too (`automodel="StableDiffusionImg2ImgPipeline"`).

### Precision and placement

Precision goes by **`torch_dtype=`**, diffusers' spelling, since the keyword is forwarded to `DiffusionPipeline.from_pretrained` untouched. `dtype=`, which is what `TransformersModel` takes, is accepted and ignored:

```python
DiffusionModel(REPO, torch_dtype=torch.float16, dispatch=True).unet.dtype   # torch.float16
DiffusionModel(REPO, dtype=torch.float16, dispatch=True).unet.dtype         # torch.float32
```

Placement follows diffusers too, and is the reverse of `TransformersModel`: `device_map="cuda"` puts the pipeline on the GPU, and `device=` is accepted and ignored. With neither, the pipeline loads on CPU.

```python
DiffusionModel(REPO, device_map="cuda", dispatch=True).unet.device   # cuda:0
DiffusionModel(REPO, device="cuda", dispatch=True).unet.device       # cpu
```

## Canonical pattern

`.trace()` and `.generate()` both run the **whole pipeline**, and `model.output` (like `tracer.result`) is the pipeline's return object (with `.images`).

!!! warning "A traced run denoises once by default"

    `.trace()` and `with model.generate(...):` both default to `num_inference_steps=1` — a fast probe pass. Only a bare `model.generate(...)` call, outside a trace, uses the pipeline's own default. A single step gives a noise blob rather than an image, with nothing printed to say so.

    **Pass `num_inference_steps=N` whenever the image itself matters.**

```python
from nnsight import DiffusionModel

sd = DiffusionModel("hf-internal-testing/tiny-stable-diffusion-torch")

with sd.generate("a photo of a cat", num_inference_steps=2, output_type="np") as tracer:
    out = sd.output.save()

print(out.images.shape)        # (1, 128, 128, 3)
```

Run without a trace to bypass NNsight entirely:

```python
out = sd.generate("a photo of a cat", num_inference_steps=2)   # default output_type -> PIL
out.images[0].save("cat.png")
```

### Reading the denoiser

```python
with sd.generate("a cat", num_inference_steps=2, output_type="np"):
    unet_out = sd.unet.output[0].save()      # unet runs return_dict=False -> tuple
```

Transformer-based pipelines (Flux, DiT, SD3) use `sd.transformer` instead of `sd.unet`.

### Running one component's forward alone

To run a single component (rather than the whole pipeline), trace that envoy directly. This needs a dispatched model — a child-envoy trace does not dispatch the model for you:

```python
sd.dispatch()

# build one denoiser forward's inputs from its config
cfg = sd.unet.config
sample = torch.randn(1, cfg.in_channels, cfg.sample_size, cfg.sample_size)
timestep = torch.tensor(1.0)
encoder_hidden_states = torch.randn(1, 4, cfg.cross_attention_dim)

with sd.unet.trace(sample, timestep, encoder_hidden_states=encoder_hidden_states):
    out = sd.unet.output.save()
# out.sample.shape == sample.shape   # UNet2DConditionOutput
```

### Multi-prompt / batched generation

Multiple `tracer.invoke(prompt)` blocks batch into a single pipeline run. Each invoke's interventions on the denoiser (`sd.unet` / `sd.transformer`) are narrowed to just that invoke's rows — accounting for classifier-free-guidance doubling and `num_images_per_prompt`. Open the trace with the shared kwargs (no top-level prompt) and give each prompt its own `invoke`:

```python
with sd.generate(num_inference_steps=2, output_type="np") as tracer:
    with tracer.invoke("a cat"):
        a = sd.unet.output[0].save()
    with tracer.invoke("a dog"):
        b = sd.unet.output[0].save()
# a.shape[0] == 2 and b.shape[0] == 2  (each invoke sees its own uncond + cond rows)
```

With `num_images_per_prompt=2`, each invoke's denoiser view is `2 images x guidance = 4` rows. Edits stay isolated to their invoke — zeroing one prompt's denoiser rows leaves the other prompt's rows untouched.

The *pipeline result object* cannot be read per invoke in a batched trace; read component tensors like `sd.unet.output[0]` inside each invoke.

### Iterating across denoising steps

The denoiser runs once per iteration of the pipeline's loop, and `tracer.iter[:N]` repeats the block's body for the first `N` of them. Bound the loop by the number of **denoiser calls**, which the scheduler decides — it is not always `num_inference_steps`:

```python
import nnsight

STEPS = 2
sd.pipeline.scheduler.set_timesteps(STEPS)
CALLS = len(sd.pipeline.scheduler.timesteps)      # PNDM: 3 for 2 steps

with sd.generate("a cat", num_inference_steps=STEPS, output_type="np") as tracer:
    outs = nnsight.save([])
    for step in tracer.iter[:CALLS]:
        outs.append(sd.unet.output[0])
    images = tracer.result.save()
# len(outs) == CALLS; step is a plain int, so `if step > 3:` really branches
```

SD 1.x defaults to `PNDMScheduler`, which calls the denoiser `num_inference_steps + 1` times (50 steps, 51 forwards); `DDIMScheduler` on the same pipeline calls it exactly `num_inference_steps` times. `len(scheduler.timesteps)` after `set_timesteps` is the count in both cases.

Bounding the loop past the calls the run makes cuts it short with a warning — bounded and open loops alike — and the statements after the loop never run: `tracer.result.save()` among them, leaving the name unbound after the block. If the loop body touches no module at all, an open `tracer.iter[:]` never returns.

### Intervening / skipping

```python
# zero the UNet output
with sd.generate("a cat", num_inference_steps=2, output_type="np"):
    sd.unet.output[0][:] = 0
    zeroed = sd.output.save()

# bypass a component with a replacement value: skip() takes what that component
# would have returned, so build it from the config
cfg = sd.unet.config
shape = (2, cfg.block_out_channels[0], cfg.sample_size, cfg.sample_size)  # 2 rows = guidance
with sd.generate("a cat", num_inference_steps=2, output_type="np", seed=0) as tracer:
    sd.unet.conv_in.skip(torch.zeros(shape))
    skipped = tracer.result.save()
```

### Caching component activations

```python
with sd.generate("a cat", num_inference_steps=2, output_type="np") as tracer:
    cache = tracer.cache(modules=[sd.unet]).save()
```

### Reproducibility

Pass `seed=` (an int); it becomes a reproducible `generator`. For a batch (`num_images_per_prompt>1` or several prompts) it fans out to one generator per image (`seed + i`), so each image is independently reproducible. Diffusers' native `generator=` still works and overrides `seed=`.

```python
a = sd.generate("a cat", seed=7, num_inference_steps=2, output_type="np")
b = sd.generate("a cat", seed=7, num_inference_steps=2, output_type="np")
# a.images == b.images
```

Renaming components (`rename={"unet": "denoiser"}`) works as on any wrapper; see [rename-modules.md](../usage/rename-modules.md).

## Special properties

| Attribute | Description | Source |
|-----------|-------------|--------|
| `model.pipeline` | The raw `diffusers.DiffusionPipeline`. Use it for non-traced operations (scheduler swaps, `save_pretrained`, ...). | `DiffusionModel._load` |
| `model.unet` / `model.transformer` | The denoiser envoy (attribute name depends on the pipeline). | `_PipelineModule.__init__` |
| `model.vae`, `model.text_encoder` (and `_2` for SDXL/Flux) | Other `nn.Module` components, as envoys. | same |
| `model.output` | The pipeline's return object (e.g. `.images`) inside a trace. | — |
| `model._module` | The `_PipelineModule` wrapper; `model._module.pipeline` is the same object as `model.pipeline`. | `_PipelineModule` |
| `model.dispatched` | Whether real weights are loaded. | `MetaMixin` |

Only `torch.nn.Module` components are envoys — the **scheduler is not**. Swap or configure it on `model.pipeline` before the trace.

## Limitations

- **Writes inside the denoising loop are ordered.** Reading a later component and then writing an earlier one raises `OutOfOrderError` rather than parking the write until the next step, whether or not the access sits inside a `tracer.iter` loop.
- **Op-level `.source` raises `SourceNotAvailable`** only when a forward has no Python source to instrument (a builtin or C function); see [source.md](../usage/source.md).
- **Remote:** diffusion models are not deployed on NDIF.

## Related

- [docs/models/index.md](index.md) — decision tree
- [docs/models/nnsight-base.md](nnsight-base.md) — base wrapper
- `src/nnsight/modeling/diffusion.py` — source (`_PipelineModule`, `DiffusionModel`, `DiffusionBatcher`, meta build)
- `tests/test_diffusion.py` — runnable examples (build, generate/trace, component trace, interventions, iteration, skip, cache, rename, seed reproducibility, batching, output types)

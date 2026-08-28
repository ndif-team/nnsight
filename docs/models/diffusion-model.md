---
title: DiffusionModel
one_liner: Wraps any diffusers DiffusionPipeline (UNet- or transformer-based) with NNsight tracing.
tags: [models, diffusion, diffusers]
related: [docs/models/index.md, docs/models/nnsight-base.md, docs/models/transformers-model.md]
sources: [src/nnsight/modeling/diffusion.py:38, src/nnsight/modeling/huggingface.py:15, tests/test_diffusion.py]
---

# DiffusionModel


## What this is for

`nnsight.DiffusionModel` wraps any `diffusers.DiffusionPipeline` so you can trace and intervene on its sub-modules — UNet (Stable Diffusion), transformer (Flux, DiT, SD3), VAE, text encoder — with the same NNsight API as other models. It supports:

- Running the whole pipeline via `.trace()` (defaults to a fast one-step pass) or `.generate()` (the pipeline's default step count)
- Iterating across denoising steps with `tracer.iter[:]`
- Multi-prompt / batched generation via `tracer.invoke(...)`, with denoiser interventions narrowed per-invoke
- Reproducible runs with `seed=` (or diffusers' native `generator=`)
- Lazy meta-tensor loading — only config files are downloaded with `dispatch=False`
- Both UNet-based pipelines (SD 1.x/2.x/XL) and transformer-based pipelines (Flux, DiT, SD3)

## When to use / when not to use

Use `DiffusionModel` when:
- You have a diffusers pipeline (`StableDiffusionPipeline`, `FluxPipeline`, `DiTPipeline`, etc.) loadable from a HuggingFace repo.
- You want to study denoising trajectories, intervene on the UNet / transformer / VAE / text encoder, or capture activations across inference steps.

Do not use `DiffusionModel` when:
- Your pipeline isn't a `diffusers.DiffusionPipeline` — wrap the underlying `torch.nn.Module` with [`NNsight`](nnsight-base.md).
- You need vLLM-style serving — vLLM is for LLMs, not diffusion pipelines.

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
    revision=None,          # git branch / tag / commit
    dispatch=False,         # True = load real weights now; False = lazy meta build
    rename=None,            # dict of module-path aliases
    **kwargs,               # forwarded to DiffusionPipeline.from_pretrained()
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | HuggingFace repo id (e.g. `"stabilityai/stable-diffusion-2-1"`, `"black-forest-labs/FLUX.1-schnell"`). |
| `dispatch` | `True` loads real weights via `DiffusionPipeline.from_pretrained` during `__init__`. `False` (default) builds a meta pipeline: each `nn.Module` component from its config on the `meta` device, light components (schedulers, tokenizers) loaded normally (`diffusers.py:60`). |
| `rename` | Module-path aliases, e.g. `{"unet": "denoiser"}`. |
| `torch_dtype`, `safety_checker=None`, `variant`, ... | Forwarded to `from_pretrained` for real loading. |

The concrete pipeline class is resolved automatically from the repo's `model_index.json` (`_class_name`) — there is **no** `automodel=` parameter (that was in old nnsight).

## Canonical pattern

`.trace()` and `.generate()` both run the **whole pipeline**, and `model.output` (like `tracer.result`) is the pipeline's return object (with `.images`). They differ only in the default step count:

- **`.trace()`** defaults to `num_inference_steps=1` — a fast one-step pass for inspecting or editing activations.
- **`.generate()`** uses the pipeline's own default step count.

Either way, pass `num_inference_steps=N` to override.

```python
from nnsight import DiffusionModel

sd = DiffusionModel("hf-internal-testing/tiny-stable-diffusion-torch")

with sd.generate("a photo of a cat", num_inference_steps=2, output_type="np") as tracer:
    out = sd.output.save()

print(out.images.shape)        # (1, 128, 128, 3)
```

`.trace()` is the same but one-step unless you say otherwise:

```python
with sd.trace("a photo of a cat", output_type="np"):   # one denoising step
    out = sd.output.save()
# out.images.shape == (1, 128, 128, 3)
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

Note: reading the *pipeline result object* per-invoke inside a batched trace is not supported (the required-field `StableDiffusionPipelineOutput` can't be rebuilt by the row-narrowing walk). Batched interventions read component tensors like `sd.unet.output[0]`.

### Iterating across denoising steps

```python
import nnsight

with sd.generate("a cat", num_inference_steps=2, output_type="np") as tracer:
    outs = nnsight.save([])
    for _ in tracer.iter[:]:
        outs.append(sd.unet.output[0])
# outs has at least one entry per inference step
```

### Intervening / skipping

```python
# zero the UNet output
with sd.generate("a cat", num_inference_steps=2, output_type="np"):
    sd.unet.output[0][:] = 0
    zeroed = sd.output.save()

# bypass a component with a replacement value
with sd.generate("a cat", num_inference_steps=2, output_type="np"):
    sd.unet.conv_in.skip(torch.zeros_like(conv_in_shape))
```

### Caching component activations

```python
with sd.generate("a cat", num_inference_steps=2, output_type="np") as tracer:
    cache = tracer.cache(modules=[sd.unet]).save()
```

### Reproducibility

Pass `seed=` (an int) — it is turned into a reproducible `generator` internally:

```python
a = sd.generate("a cat", seed=7, num_inference_steps=2, output_type="np")
b = sd.generate("a cat", seed=7, num_inference_steps=2, output_type="np")
# a.images == b.images
```

For a batch (`num_images_per_prompt>1` or multiple prompts) a single `seed` fans out to one generator per image (`seed + i`), so each image is independently reproducible while the run as a whole stays deterministic:

```python
out = sd.generate("a cat", seed=7, num_images_per_prompt=2, num_inference_steps=2, output_type="np")
# out.images.shape[0] == 2; the two images differ, but re-running with seed=7 reproduces both
```

Passing diffusers' native `generator=` still works and **overrides** `seed=`:

```python
g = torch.Generator().manual_seed(7)
out = sd.generate("a cat", generator=g, num_inference_steps=2, output_type="np")
```

### Renaming components

```python
model = DiffusionModel(REPO, rename={"unet": "denoiser"})
with model.generate("a cat", num_inference_steps=2, output_type="np"):
    denoised = model.denoiser.output[0].save()
```

## Special properties

| Attribute | Description | Source |
|-----------|-------------|--------|
| `model.pipeline` | The raw `diffusers.DiffusionPipeline`. Use it for non-traced operations (scheduler swaps, `save_pretrained`, ...). | `diffusers.py:52` |
| `model.unet` / `model.transformer` | The denoiser envoy (attribute name depends on the pipeline). | `_PipelineModule.__init__` |
| `model.vae`, `model.text_encoder` (and `_2` for SDXL/Flux) | Other `nn.Module` components, as envoys. | same |
| `model.output` | The pipeline's return object (e.g. `.images`) inside a trace. | — |
| `model._module` | The `_PipelineModule` wrapper; `model._module.pipeline` is the same object as `model.pipeline`. | `diffusers.py:17` |
| `model.dispatched` | Whether real weights are loaded. | `MetaMixin` |

Only `torch.nn.Module` components are wrapped as envoys — the **scheduler is not**. To change scheduler behavior, mutate `model.pipeline` directly before the trace.

## Limitations

- **`.trace()` defaults to `num_inference_steps=1`; `.generate()` uses the pipeline's default** (which may be large). Pass `num_inference_steps=N` to either to override.
- **Per-invoke pipeline result objects aren't supported in a batched trace.** Read component tensors (e.g. `sd.unet.output[0]`) inside `tracer.invoke(...)` blocks, not `sd.output`.
- **Scheduler / non-module components are not Envoy-wrapped.** Do scheduler swaps on `model.pipeline` pre-trace.
- **Op-level `.source` raises `SourceNotAvailable`** only when a forward has no Python source to instrument (a builtin or C function); decorated and closure forwards are instrumented — see [source.md](../usage/source.md).
- **UNet output shape varies.** With `return_dict=False` it's a tuple; use `sd.unet.output[0]`.
- **Remote:** diffusion models are not deployed on NDIF as of this writing.

## Related

- [docs/models/index.md](index.md) — decision tree
- [docs/models/nnsight-base.md](nnsight-base.md) — base wrapper
- `src/nnsight/modeling/diffusion.py` — source (`_PipelineModule`, `DiffusionModel`, `DiffusionBatcher`, meta build)
- `tests/test_diffusion.py` — runnable examples (build, generate/trace, component trace, interventions, iteration, skip, cache, rename, seed reproducibility, batching, output types)

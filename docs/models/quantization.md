---
title: Quantization
one_liner: Hold a model's weights in 4 or 8 bits by naming the format where you would name a dtype — activations, module paths, and traces are unchanged.
tags: [models, transformers, quantization, bitsandbytes, memory, 4-bit, 8-bit]
related: [docs/models/transformers-model.md, docs/models/tensor-parallel.md, docs/models/index.md, docs/remote/index.md]
sources: [src/nnsight/modeling/quantization.py, src/nnsight/modeling/mixins/remotable.py, tests/test_quantization.py]
---

# Quantization

## What this is for

A model too big for the GPU you have can be held in **fewer bits per weight**.
Name the format in the dtype slot; there is no quantizer config to build:

```python
from nnsight import TransformersModel

model = TransformersModel(
    "meta-llama/Llama-3.2-3B",
    task="text-generation",
    dtype="nf4",          # <- where you would write "bfloat16"
    dispatch=True,
)
```

`bitsandbytes` and `accelerate` must be installed; see [Requirements](#requirements).

## The names

| Name | What you get | Bytes/weight |
|---|---|---|
| `nf4`, `int4`, `4bit` | bitsandbytes 4-bit, NF4 | 0.5 |
| `fp4` | bitsandbytes 4-bit, FP4 | 0.5 |
| `int8`, `8bit` | bitsandbytes LLM.int8() | 1.0 |
| `fp8` | transformers block-wise FP8, compute capability 8.9+ | 1.0 |

The unqualified 4-bit names mean **NF4**, which bitsandbytes recommends and which
measures better than `fp4` at the same size (see [What it costs](#what-it-costs)).

> **`fp8` needs a GPU of compute capability 8.9 or better** (4090, L40S, H100;
> not A100 at 8.0). Below it, transformers itself only logs a warning and loads
> bfloat16 while still reporting the model as quantized; nnsight raises a
> `ValueError` instead, before any weights are fetched.

## What a trace sees

**The module tree is identical.** A quantized linear is a different class holding
a differently-shaped weight, but it sits at the same path with the same children,
so every intervention, envoy and remote request naming a module is unaffected.

**Activations are ordinary 16-bit tensors.** The weights are narrow; what flows
between modules is not. `layers[5].mlp.gate_proj.output` comes back the same
shape and dtype it would from a `bfloat16` model.

```python
with model.trace("The Eiffel Tower is in the city of"):
    gate = model.model.layers[5].mlp.gate_proj.output.save()
    logits = model.lm_head.output.save()

print(gate.shape, gate.dtype)   # (1, 11, 8192) torch.bfloat16
```

A **float32** checkpoint is the case where that sentence bites. Quantizing it
moves the whole model to the compute dtype, so activations that were float32
arrive at half the width: GPT-2's `c_attn.output` is `torch.float32`
unquantized and `torch.bfloat16` at `nf4`, norms and LM head included.

**Raw weights are the exception.** `gate_proj.weight` under 4-bit is a packed
`uint8` blob — `(8388608, 1)` where the real matrix is `(8192, 2048)` — because
that is genuinely how it is stored. Read activations, not weights, or load
unquantized when the weights are the object of study. Parameter counts follow the
storage rather than the model: `sum(p.numel() for p in model._module.parameters())`
gives 749,275,136 for a 4-bit Llama-3.2-1B against 1,235,814,400 unquantized.

## What it costs

Measured on Llama-3.2-1B over the 86 next-token distributions of a fixed
passage. **KL** is against the same model at `bfloat16`, and **top-1
agreement** is how often the two pick the same next token:

| dtype | weights | vs bf16 | mean KL | max KL | top-1 agreement |
|---|---|---|---|---|---|
| `bfloat16` | 2.30 GB | 1.00x | — | — | — |
| `int8` | 1.40 GB | 0.61x | 0.011 | 0.09 | 95.3% |
| `nf4` | 1.00 GB | 0.43x | 0.143 | 1.64 | 87.1% |
| `fp4` | 1.00 GB | 0.43x | 0.182 | 1.37 | 82.4% |

**4-bit changes the argmax next token 13% of the time**, so treat a quantized run
as a different model rather than a cheaper copy of the same one, and do not
compare activations across widths. `fp4` disagrees more often than `nf4` at the
same size, which is why the unqualified names point at NF4.

A hidden-state norm is not a substitute for either column. Llama-3.2-1B's
layer-5 norm moves from 423.9 to 410.0 at `nf4`, about 3%, while the argmax under
it changes on one token in eight.

### Speed

Quantized weights are unpacked on the way into each matmul, so a quantized
forward is slower: `nf4` is a little slower than `bfloat16`, and **`int8` is
several times slower than `nf4`**. `int8` is the accurate format and the slow
one at once; for a sweep over hundreds of forwards, `nf4` costs accuracy you
can measure and buys back most of the time.

> **Sizing runs low.** A model does not shrink by the ratio in the names table:
> embeddings, norms and the LM head stay 16-bit in every format here, and the
> format stores a scale per block. Llama-3.2-1B at `nf4` takes 1.00 GB where 0.5
> bytes/weight predicts 0.58; counting the embeddings at 2 bytes and the rest at
> the format's width predicts 0.94. Measure when it matters.

## Compute dtype

The compute dtype — what unquantized modules hold and what activations arrive
as — is `bfloat16`, except that **`int8` computes in `float16`** (bitsandbytes
implements LLM.int8() that way and casts anything else, warning once per matmul). Override it if you need to:

```python
TransformersModel(repo_id, task="text-generation", dtype="nf4", compute_dtype="float32")
```

### Gradients through an `int8` model

float16's narrow exponent overflows on a backward over a large-magnitude loss.
On GPT-2, `logits.sum()` as the loss gives an all-NaN gradient at `int8`:

```python
with model.trace("The Eiffel Tower is in the city of"):
    a = model.transformer.h[0].output
    loss = model.output.logits.sum()
    with loss.backward():
        grad = a.grad.save()

print(grad.dtype, float(grad.float().norm()))   # torch.float16 nan
```

Nothing raises; the NaNs flow into whatever uses the gradient (for attribution
patching, the score itself). Use a loss on a scale float16 can hold (a
log-probability of one token), or use `nf4`, which computes in bfloat16.

## What the format does not touch

bitsandbytes swaps `nn.Linear` and nothing else, which decides what a given
checkpoint saves.

**A mixture-of-experts model barely shrinks.** transformers 5 holds the experts
as stacked 3-D parameters on one module (`Qwen3MoeExperts` and its equivalents)
rather than as linears, so bitsandbytes leaves them at the compute dtype and
quantizes only the attention projections, the router and the shared layers.
Those are the minority of an MoE's weights. Quantizing Qwen1.5-MoE-A2.7B at
`nf4` takes it from 12.89 GiB to 12.53 GiB, under 3%, while still perturbing
every routing decision the attention output feeds. Tensor parallelism is the
answer for an MoE that does not fit; see
[tensor-parallel.md](tensor-parallel.md).

## Requirements

`pip install bitsandbytes accelerate`. Neither is a dependency of nnsight, and
transformers raises `ImportError` naming whichever is missing when the load
reaches the quantizer.

A GPU is not required: bitsandbytes quantizes and runs `nf4` and `int8` on CPU.
What the quantizers reject is the `meta` device.

That refusal is why `dispatch=False` is still fine: the **meta build ignores the
quantization** and builds the architecture at the compute dtype, which is what
makes the lazy path work and what lets a client model a checkpoint a server holds
quantized. The weights are only quantized when they are actually loaded.

## Names that are not formats

**A torch dtype nothing can load falls back to float32.** `torch.int1` through
`torch.int7` exist, so `dtype="int3"` is not rejected as a name; transformers
fails to load it, logs `Falling back to torch.float32 ...`, and loads a float32
model — twice the bfloat16 size. Nothing raises. (With `dispatch=False` the meta
build does raise `ValueError: ... not a floating-point dtype`.)

**`load_in_4bit=` and `load_in_8bit=` are not transformers 5 arguments.** Most
tutorials written against transformers 4 pass them; in transformers 5 they reach the
model class as a stray kwarg, and the pipeline reports it fifteen lines into a
nested traceback:

```
ValueError: Could not load model meta-llama/Llama-3.2-1B with any of the following classes: (...)
  ...
  TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'
```

Use `dtype="nf4"` or `dtype="int8"`. A `quantization_config=BitsAndBytesConfig(...)`
of your own also still works, though not together with a quantization name:
passing both raises, since they are two answers to how the weights are held.

## On NDIF

A **client cannot ask for a quantization.** A remote model key is the repo id,
revision, task and `trust_remote_code`, and says nothing about how the weights are held; the
deployment decides. A client-side `dtype=` only shapes its own meta build.

## Related

- [transformers-model.md](transformers-model.md) — the wrapper being quantized.
- [tensor-parallel.md](tensor-parallel.md) — the other way to fit a model that
  does not fit, by splitting it across GPUs rather than narrowing it. Numerically
  faithful, and the only one of the two that helps an MoE.

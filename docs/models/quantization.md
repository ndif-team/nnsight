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
That normally means building a quantizer config and knowing which of
transformers' several backends your format belongs to — a lot of ceremony for a
choice that is really just *how wide is a weight*. So it goes in the dtype slot,
next to the widths torch does have:

```python
from nnsight import TransformersModel

model = TransformersModel(
    "meta-llama/Llama-3.2-3B",
    task="text-generation",
    dtype="nf4",          # <- where you would write "bfloat16"
    dispatch=True,
)
```

Nothing else changes. There is nothing to import and no config to build.

## The names

| Name | What you get | Bytes/weight |
|---|---|---|
| `nf4`, `int4`, `4bit` | bitsandbytes 4-bit, NF4 | 0.5 |
| `fp4` | bitsandbytes 4-bit, FP4 | 0.5 |
| `int8`, `8bit` | bitsandbytes LLM.int8() | 1.0 |
| `fp8` | transformers block-wise FP8 (H100+) | 1.0 |

Several names for one thing on purpose: someone reaching for 4-bit writes
whichever of `int4` / `4bit` / `nf4` they last read about, and there is nothing
gained by making two of the three an error. The unqualified names mean **NF4**,
which is what bitsandbytes recommends; `fp4` is reached only by asking for it.

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

print(gate.shape, gate.dtype)   # (1, 11, 8192) torch.bfloat16 — as if unquantized
```

**Raw weights are the exception.** `gate_proj.weight` under 4-bit is a packed
`uint8` blob — `(8388608, 1)` where the real matrix is `(8192, 2048)` — because
that is genuinely how it is stored. Read activations, not weights, or load
unquantized when the weights are the object of study.

## What it costs

Measured on Llama-3.2-1B, layer-5 hidden-state norm against the unquantized run:

| dtype | GPU | norm | next token |
|---|---|---|---|
| `bfloat16` | 2.47 GB | 422.17 | ` Paris` |
| `int8` | 1.50 GB | 422.07 | ` Paris` |
| `nf4` | 1.07 GB | 408.76 | ` Paris` |
| `fp4` | 1.07 GB | 384.41 | ` Paris` |

4-bit is a real perturbation — a few percent on hidden-state norms, growing with
depth — so treat a quantized run as a different model rather than a cheaper copy
of the same one, and do not compare activations across widths.

> **Sizing runs low.** A model does not shrink by the ratio in the table: the
> format leaves embeddings, norms and the LM head in 16 bits and stores a scale
> per block. Llama-3.2-1B at `nf4` really takes 1.07 GB where 0.5 bytes/weight
> predicts 0.62. Budget from measurement, not arithmetic.

## Compute dtype

Everything the format does not quantize — norms, embeddings, the LM head — and
everything the model computes in is `bfloat16`, with one exception: **`int8`
computes in `float16`**, because bitsandbytes implements LLM.int8() that way and
casts anything else on the way in, warning once per matmul as it does. So an
`int8` model's activations arrive as `float16`.

Override it if you need to:

```python
TransformersModel(repo_id, task="text-generation", dtype="nf4", compute_dtype="float32")
```

## Requires a GPU and the backend

bitsandbytes needs a CUDA device — a quantized model cannot be loaded on CPU or
built on meta. `dispatch=False` is still fine: the **meta build ignores the
quantization** and builds the architecture at the compute dtype, which is what
makes the lazy path work and what lets a client model a checkpoint a server holds
quantized. The weights are only quantized when they are actually loaded.

`fp8` is not bitsandbytes at all — it is transformers' own quantizer and needs
H100-or-later hardware, which transformers checks for.

## On NDIF

The same names are what a deployment is configured with, so a server can hold a
model 4-bit without anything client-side changing:

```
ndif deploy meta-llama/Llama-3.3-70B --dtype nf4
```

A **client cannot ask for a quantization**, though — a remote model key is the
repo id and revision, and says nothing about how the weights are held. The
deployment decides; a client-side `dtype=` only shapes its own meta build.

## Related

- [transformers-model.md](transformers-model.md) — the wrapper being quantized.
- [tensor-parallel.md](tensor-parallel.md) — the other way to fit a model that
  does not fit, by splitting it across GPUs rather than narrowing it.

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

`bitsandbytes` and `accelerate` do have to be installed; neither comes with
nnsight. See [Requirements](#requirements).

## The names

| Name | What you get | Bytes/weight |
|---|---|---|
| `nf4`, `int4`, `4bit` | bitsandbytes 4-bit, NF4 | 0.5 |
| `fp4` | bitsandbytes 4-bit, FP4 | 0.5 |
| `int8`, `8bit` | bitsandbytes LLM.int8() | 1.0 |
| `fp8` | transformers block-wise FP8, compute capability 8.9+ | 1.0 |

Several names for one thing on purpose: someone reaching for 4-bit writes
whichever of `int4` / `4bit` / `nf4` they last read about, and there is nothing
gained by making two of the three an error. The unqualified names mean **NF4**,
which is what bitsandbytes recommends; `fp4` is reached only by asking for it,
and measures worse than `nf4` at the same size (see [What it
costs](#what-it-costs)).

> **`fp8` loads unquantized on a GPU below compute capability 8.9.** It is
> transformers' own quantizer rather than bitsandbytes, and on older hardware it
> does not refuse: it logs a warning, sets `dequantize` on the quantization
> config, and loads bfloat16 instead. The load succeeds, and the
> `FineGrainedFP8HfQuantizer` object stays attached, so a model that checks
> `hf_quantizer` or `config.quantization_config` is told it is quantized.
> Measured on an A6000 (8.6), Llama-3.2-1B at `fp8` is bit-identical to the same
> model at `bfloat16`: the same 2.30 GB of weights, `Linear` rather than
> `FP8Linear`, and a KL of 0.000000 against the bfloat16 run. 4090 and L40S
> (both 8.9) qualify; A100 (8.0) does not. What tells you which you got is
> `model._module.config.quantization_config.dequantize` — `True` means the
> weights are bfloat16.

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

Measured on Llama-3.2-1B, one RTX A6000, over the 86 next-token distributions of
a fixed passage. **KL** is against the same model at `bfloat16`, and **top-1
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

The damage shrinks as the model grows. The same measurement on Llama-3.1-8B
(14.96 GB bfloat16) gives `nf4` 5.65 GB at mean KL 0.049 and **92.9%** top-1
agreement, and `int8` 8.46 GB at KL 0.010 and 96.5%.

A hidden-state norm is not a substitute for either column. Llama-3.2-1B's
layer-5 norm moves from 423.9 to 410.0 at `nf4`, about 3%, while the argmax under
it changes on one token in eight.

### Speed

Quantized weights are unpacked on the way into each matmul, so a quantized
forward is slower. Llama-3.2-1B, mean of 20 forwards on an 11-token prompt, over
four runs on a shared A6000:

| dtype | forward | vs bf16 |
|---|---|---|
| `bfloat16` | 14–18 ms | 1.00x |
| `nf4` | 20–24 ms | 1.1–1.7x |
| `int8` | 50–67 ms | 2.9–4.5x |

Ranges rather than points because the card is shared and the ratio moved by that
much between runs. What did not move is the ordering: `int8` was at least 2.9x
`bfloat16` in every run. It is the accurate format and the slow one at once, and
for a sweep over hundreds of forwards that is usually what decides between the
two — `nf4` costs accuracy you can measure and buys back most of the time.

> **Sizing runs low.** A model does not shrink by the ratio in the names table:
> the format leaves embeddings, norms and the LM head in 16 bits and stores a
> scale per block. Llama-3.2-1B at `nf4` really takes 1.00 GB where 0.5
> bytes/weight predicts 0.58. Counting the embeddings at 2 bytes and the rest at
> the format's width predicts 0.94, which is close enough to budget from. Better
> still, measure.

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

### Gradients through an `int8` model

float16 carries a much narrower exponent than bfloat16, and a backward pass over
a large-magnitude loss runs out of it. On GPT-2, `logits.sum()` as the loss gives
a gradient that is entirely NaN at `int8`, where the unquantized model reports a
norm of 934,879 and `nf4` reports 1,079,000:

```python
with model.trace("The Eiffel Tower is in the city of"):
    a = model.transformer.h[0].output
    loss = model.output.logits.sum()
    with loss.backward():
        grad = a.grad.save()

print(grad.dtype, float(grad.float().norm()))   # torch.float16 nan
```

Nothing raises; the NaNs propagate into whatever you compute from the gradient,
which for attribution patching is the attribution score itself. A loss on a
scale float16 can hold fixes it — a log-probability of one token gives 0.52 at
`int8` against 0.59 at `nf4` — and so does using `nf4`, which computes in
bfloat16 and survives either loss.

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

Embeddings, norms and the LM head also stay 16-bit in every format here, which
is most of the gap between the arithmetic and the measurement above.

## Requirements

`pip install bitsandbytes accelerate`. Neither is a dependency of nnsight, and
transformers raises `ImportError` naming whichever is missing when the load
reaches the quantizer.

A GPU is not required. bitsandbytes 0.50 quantizes and runs on CPU: with
`CUDA_VISIBLE_DEVICES=""`, `dtype="nf4"` gives a `Linear4bit` whose weight lives
on the CPU, and a forward that reaches the same layer-5 norm as the GPU's to five
significant figures (408.75) and predicts the same token. `int8` loads on CPU
too. What the quantizers do reject is the `meta` device.

That refusal is why `dispatch=False` is still fine: the **meta build ignores the
quantization** and builds the architecture at the compute dtype, which is what
makes the lazy path work and what lets a client model a checkpoint a server holds
quantized. The weights are only quantized when they are actually loaded.

## Names that are not formats

**A torch dtype nothing can load falls back to float32.** `torch.int1` through
`torch.int7` exist, so `dtype="int3"` is not rejected as a name; transformers
tries it, fails, logs `Falling back to torch.float32 because loading with the
original dtype failed on the target device`, and loads a float32 model. On
Llama-3.2-1B that is 4.60 GB, twice the bfloat16 default, in answer to a request
for something narrower. Nothing raises. (The meta path does raise — `dispatch=False`
with `dtype="int3"` gives `ValueError: ... cannot be instantiated under
dtype=torch.int3 as it's not a floating-point dtype`.)

**`load_in_4bit=` and `load_in_8bit=` are not transformers 5 arguments.** They
are what most tutorials written against transformers 4 pass, and what an LLM
asked for 4-bit loading will usually write. In transformers 5 they reach the
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

The same names are what a deployment is configured with, so a server can hold a
model 4-bit without anything client-side changing:

```
ndif deploy meta-llama/Llama-3.3-70B --dtype nf4
```

A **client cannot ask for a quantization**, though — a remote model key is the
repo id and revision, and says nothing about how the weights are held. The
deployment decides; a client-side `dtype=` only shapes its own meta build.

Placement uses the nominal bytes/weight from the names table, which undercounts
by the same margin shown above, so a quantized deployment needs padding beyond
NDIF's default 0.15.

## Related

- [transformers-model.md](transformers-model.md) — the wrapper being quantized.
- [tensor-parallel.md](tensor-parallel.md) — the other way to fit a model that
  does not fit, by splitting it across GPUs rather than narrowing it. Numerically
  faithful, and the only one of the two that helps an MoE.

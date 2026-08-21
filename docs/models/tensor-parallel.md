---
title: Tensor Parallelism
one_liner: Trace a model sharded across GPUs with transformers tensor parallelism — sharded activations are gathered so a trace reads exactly as it would on one GPU.
tags: [models, transformers, tensor-parallel, multi-gpu, distributed]
related: [docs/models/transformers-model.md, docs/models/vllm.md, docs/models/index.md, docs/usage/generate.md, docs/usage/cache.md]
sources: [src/nnsight/modeling/tp/interleaver.py, src/nnsight/modeling/huggingface.py, tests/test_transformers_tensor_parallel.py, tests/tp_worker.py]
---

# Tensor Parallelism

## What this is for

A model too big for one GPU can be **split across several** with transformers'
native tensor parallelism: each rank holds a slice of every attention and MLP
projection. This is different from `device_map="auto"`, which puts whole *layers*
on different GPUs and runs them one after another — tensor parallelism splits
*within* each layer and runs the ranks together.

The catch for interpretability is that a sharded module's activation, on any one
rank, is only that rank's slice of the real tensor. nnsight gathers those slices
before your intervention sees the value and re-splits whatever you leave behind,
so **the trace you write is the trace you would write against one GPU**.

There is nothing to install, import, or enable.

## The canonical pattern

Tensor parallelism needs one process per GPU, so the script is launched with
`torchrun` (or `python -m torch.distributed.run`) and **every rank runs the whole
script, including your intervention code**.

```python
# tp_trace.py  —  torchrun --nproc_per_node=4 tp_trace.py
import torch
from transformers.distributed import DistributedConfig
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "meta-llama/Llama-3.2-3B",
    task="text-generation",
    dispatch=True,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=4),
)

with model.trace("The Eiffel Tower is in the city of"):
    # gate_proj is column-parallel: each rank computes 2048 of these 8192
    # features. Read it and you get all 8192.
    gate = model.model.layers[5].mlp.gate_proj.output.save()
    logits = model.lm_head.output.save()

print(gate.shape)  # (1, 11, 8192) on every rank, not (1, 11, 2048)
```

Edits work the same way: you edit the whole tensor and nnsight puts each rank's
piece back.

```python
with model.trace(prompt):
    # Spans rank boundaries; you never think about that.
    model.model.layers[5].mlp.gate_proj.output[..., :3000] = 0
    logits = model.lm_head.output.save()
```

> `tp_plan="auto"` / `tp_size=` are **not** `from_pretrained` arguments in
> transformers 5.x, despite what its `from_pretrained` docstring still says. Use
> `distributed_config=DistributedConfig(tp_size=N)`.

## What is sharded and what is not

Most of what people read is **already whole** and costs nothing: a row-parallel
layer all-reduces its output, so a decoder layer, `self_attn`, `mlp`, and the
final norm all arrive complete. Only two kinds of value are really a slice:

| | Sharded? | Example |
|---|---|---|
| Column-parallel **output** | yes, gathered for you | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` |
| Row-parallel **input** | yes, gathered for you | `o_proj.input`, `down_proj.input` |
| Row-parallel output | no — all-reduced | `o_proj.output`, `down_proj.output` |
| Whole modules | no | `model.layers[i].output`, `mlp.output`, `norm.output` |
| The LM head | no — gathered by transformers | `lm_head.output` |
| Embeddings | no — all-reduced | `embed_tokens.output` |
| **Parameters** | **yes — not gathered** | `q_proj.weight`, `down_proj.weight` |

Calling a sharded module **ad hoc** — the logit lens, `model.lm_head(hidden)` —
is corrected the same way its activations are, so it returns the full-width
result it would on one GPU.

Parameters are the exception, and the one place a trace does not read as it would
on one GPU: `layer.weight` is this rank's slice, at `1/tp_size` of the real
width. Weights are what tensor parallelism exists to split, so nnsight does not
quietly reassemble one — that would allocate the whole tensor on every rank, in
the situation where memory was tight enough to reach for TP. Gather it yourself
if you need it whole, and remember every rank must do so.

The gather only fires when an intervention is actually parked on that location,
so reading a handful of locations does not pay for the hundreds you ignored. A
`tracer.cache()` gathers only the modules it selects.

## Rules for intervention code under TP

**Every rank runs your block.** That is what keeps the collectives lined up, and
it puts two obligations on the code:

1. **No rank-dependent control flow.** Nothing may branch on rank, and nothing
   may take a different path on different ranks — the ranks would stop agreeing
   on when to gather, and the run deadlocks.

2. **Seed before you sample.** This one is a correctness bug, not an
   inconsistency. If sampling diverges, the ranks generate *different tokens*,
   and then the model's own all-reduces sum activations computed from different
   sequences — the output is wrong on every rank, not merely different. Use
   greedy decoding, or seed identically on every rank:

   ```python
   torch.manual_seed(0)                      # same on every rank
   with model.generate(prompt, max_new_tokens=20) as tracer:
       out = tracer.result.save()
   ```

   Many checkpoints ship `do_sample: true` in `generation_config.json`, so this
   bites without you asking for sampling.

**Every rank produces the same saved values**, since they are computed from
gathered tensors. Print or write results from one rank, or you get N copies.

## Requires transformers >= 5.15

Below that, a checkpoint with `tie_word_embeddings=True` — Llama-3.2, Qwen2.5,
most small models — has its LM head gathered but never sharded, so logits come
back `tp_size` times too wide. The argmax still lands inside the first copy, so
nothing downstream looks wrong. nnsight raises `UnsupportedTransformersVersion`
rather than let that through.

## A checkpoint that cannot be split at all

Not every model publishes a sharding plan. gpt2 is the obvious one: its config
has no `base_model_tp_plan`, so there is nothing telling transformers which
weights to cut or along which dimension.

**transformers does not refuse this, and does not warn.** Given a plan of `None`
it shards *nothing* — `verify_tp_plan` returns early and no hooks are installed —
so every rank loads a complete copy of the weights. The model then answers
correctly off one rank while the other cards hold redundant copies. Nothing
fails; you simply paid `tp_size` GPUs for one GPU's worth of work.

nnsight raises `UnshardableCheckpoint` instead, before the weights are fetched:

```python
TransformersModel(
    "openai-community/gpt2",
    distributed_config=DistributedConfig(tp_size=2),
)
# UnshardableCheckpoint: this checkpoint cannot be split tensor-parallel, so
# tp_size=2 would load a whole copy of it onto every rank rather than a shard.
```

The same error covers a degree that does not divide evenly — `tp_size=3` on a
model that splits 8 ways — because the all-gather assumes every rank holds an
equal piece, so an uneven degree returns the wrong shape rather than running
slower. The message lists the degrees that would work.

To find out in advance, ask
[`max_tp_size`][nnsight.modeling.tp.plan.max_tp_size] for the largest degree a
config supports; every workable degree is one of its divisors, and `None` means
the model has to be spread some other way (one GPU, or `device_map` over
several).

## What is not supported

A few expert-parallel styles slice by *expert* rather than along the feature
dimension, so the gather here does not apply: `grouped_gemm`, `ep_router`,
`megamoe_*`, `moe_identity_expert`, and MLA's split kv projection
(`mla_kv_a_proj`). Loading such a model tensor-parallel raises
`UnsupportedParallelStyle` naming the module and style — deliberately, rather than
handing you a fragment of a tensor and letting you draw conclusions from it.

**Most mixture-of-experts checkpoints are fine.** `moe_tp_experts`, which Mixtral,
DeepSeek-V3, Qwen3-MoE and around twenty-five other shipped configs use,
all-reduces inside its own forward — both sides arrive whole and nothing needs
gathering.

`float`/`bfloat16` results differ slightly from a single-GPU run (relative error
around 1e-3 to 1e-2, growing with depth) because an all-reduce sums in a
different order than one big matmul. Token choices are unaffected in practice;
the test suite asserts generated ids are identical.

## Under the hood

`TPFragments` ([`nnsight.modeling.tp`][nnsight.modeling.tp]) says which locations
hold one rank's slice and how to reassemble them; the
[`Interleaver`][nnsight.intervention.interleaver.Interleaver] does the bracketing —
gather the value, serve the parked workers the whole tensor, re-split what they
leave — once per visit, and only when something is actually waiting to read it.

Every `HuggingFaceModel` is built with an ordinary interleaver carrying one of
these. It stays inert (`enabled = False`, one attribute check per location) until
it instruments a module transformers has stamped with a `_hf_tp_plan`, which is
also where it records the rules. That covers eager loading and the
meta-then-`dispatch()` path without either needing to know about it.

The same seam serves vLLM's tensor parallelism, which shards differently and
gathers with different collectives — see
[`nnsight.intervention.fragments`][nnsight.intervention.fragments].

## Related

- [transformers-model.md](transformers-model.md) — the wrapper being sharded.
- [vllm.md](vllm.md) — the other way to shard across GPUs, with its own tradeoffs
  (throughput and continuous batching, one prompt per invoke).
- [../usage/cache.md](../usage/cache.md) — `tracer.cache()`, which gathers only
  what it selects.

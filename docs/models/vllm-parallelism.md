---
title: vLLM parallelism and architectures
one_liner: Tensor parallelism is transparent to a block; mixture-of-experts and hybrid trunks, and what changes across vLLM releases.
tags: [models, vllm, tensor-parallel, moe]
related: [docs/models/vllm.md, docs/models/tensor-parallel.md, docs/models/index.md]
sources: [src/nnsight/modeling/vllm/fragments.py, src/nnsight/modeling/vllm/batching.py, tests/vllm/test_tensor_parallel.py, tests/vllm/test_moe_batching.py]
---

# Parallelism and architectures

## Tensor parallelism is transparent

```python
model = VLLM("meta-llama/Llama-3.1-8B", tensor_parallel_size=4, dispatch=True)

with model.trace("Hello", temperature=0.0):
    out = model.model.layers[16].output             # (hidden, residual), both full width
    hidden = (out[0] + out[1]).clone().save()
```

`VLLMFragments` (`fragments.py`) gathers a `ColumnParallelLinear`/`RowParallelLinear` shard into the full tensor before your intervention reads it and re-splits on write, so every rank runs the same code on the same complete tensor: `qkv_proj.output`, `gate_up_proj.output`, `o_proj.input` and `down_proj.input` read at their full width; layer outputs and `norm.output` are whole on every rank already. Parameters are not: `layer.weight` is this rank's slice. Verified in `tests/vllm/test_tensor_parallel.py`.

Rules for block code under TP: every rank runs the block, so keep control flow rank-independent; a tensor referenced from outside travels with the block to every rank; an in-block `torch.randn` agrees across ranks (vLLM seeds every worker alike). The client-side `print(model)` shows `tp_size=1` whatever you asked for — check `layer._module.weight.shape` inside the block for the real slice.

A logit lens goes through the model's own logits path, which gathers the vocab shards; calling `model.lm_head(h)` directly raises `LMHead's weights should be used in the sampler`:

```python
out = model.model.layers[20].output
h = (out[0] + out[1])[-1:]
logits = model.logits_processor(model.lm_head, model.model.norm(h))   # [1, vocab]
top1 = logits.argmax(-1).item()
```

## Mixture-of-experts models and expert parallelism

MoE models (Qwen-MoE, DeepSeek, Mixtral, ...) work with the same transparency, in both expert layouts vLLM offers on the same ranks:

- **default** (`enable_expert_parallel=False`): every rank holds a slice of every expert's matrices (the dense-MLP TP sharding, fused across experts);
- **expert parallel** (`enable_expert_parallel=True`): each rank holds `num_experts / world_size` whole experts.

The router (`mlp.gate`, a `ReplicatedLinear`) is full and identical on every rank, so reading router logits or swapping them (expert steering / expert-masking ablation) needs no gathering at all. The fused-experts module (`mlp.experts`, a `FusedMoE`) is the one MoE-specific case the batcher handles: models that build it with `reduce_results=False` (the Qwen-MoE/DeepSeek pattern) make it return **per-rank partial sums** that the outer block all-reduces afterwards, so on access the batcher all-reduces the partials into the true value and on write-back divides by the group size so the block's own all-reduce reconstructs a swapped value exactly once. Verified in `tests/vllm/test_moe_batching.py` against an HF reference.

Individual experts are **not** addressable as submodules: vLLM stacks all local experts into fused weight tensors consumed by one grouped kernel, so there is no `experts[3]` to hook at any parallelism level. The top-k selection and routing weights are computed inside that kernel too — recompute them from the logits. `mlp.gate.output` is `(logits, bias)`, `[tokens, num_experts]`; to ablate an expert, mask its router logit: `mlp.gate.output[0][:, e] = -inf`.

```python
with model.trace(prompt, temperature=0.0, max_tokens=6) as tracer:
    tops = list().save()
    for _ in tracer.iter[:6]:
        logits, _bias = model.model.layers[5].mlp.gate.output
        tops.append(logits[-1].topk(2).indices.clone())
```

## Hybrid (linear-attention) trunks

Qwen3-Next / Qwen3.5-style models interleave gated-delta-net layers with full-attention layers. Both are ordinary decoder-layer envoys; tell them apart by the child they carry, `layers[i].linear_attn` or `layers[i].self_attn`. The recurrent state lives in vLLM's state cache, not in any module output. `taps=` works on these models: a tapped engine pins `cudagraph_mode="FULL_DECODE_ONLY"` on any model vLLM reports as hybrid or attention-free (a full graph captured over a recurrent layer silently miscomputes the other batch composition), and tapped generation matches eager exactly. Checkpoints with a vision tower (`Qwen3_5ForConditionalGeneration`) load and trace on text; their decoder layers are at `model.language_model.model.layers`.

!!! warning "This recipe is vLLM-specific"
    On **`TransformersModel`** it is a silent no-op. A transformers MoE block calls its
    router as `_, top_k_weights, top_k_index = self.gate(hidden_states)` — element `[0]`,
    the logits, is **discarded**, so masking it changes nothing (setting all 64 logits to
    `-inf` gives `max|delta| = 0.0`). There, edit the *selection* instead: write the
    routing weights or expert indices in `mlp.gate.output[1]` / `[2]`.

    Two further transformers-side traps: with `norm_topk_prob=False` masking a weight
    rescales all surviving experts, so a never-selected expert appears to have an effect
    larger than a real below-median one — ablate by selection, not by weight. And the
    router's `.input`/`.output` have **no batch axis** (`(B*T, E)`), so per-token stats
    over a padded batch silently mix in pad rows.

## vLLM versions

Tested against **0.16 through 0.27**, which is what nnsight's own suite runs on.
Two things about newer engines are worth knowing, because nnsight arranges the
first for you and cannot arrange the second:

- **The model runner.** vLLM 0.27 ships a second GPU model runner and defaults to
  it for every non-MoE model. nnsight's hooks arrive by subclassing the original
  one, so it asks vLLM for that one (`VLLM_USE_V2_MODEL_RUNNER=0`) when it builds
  the engine. Setting that variable to `1` yourself is refused rather than
  overridden — the engine would otherwise come up with no interventions installed
  and fail at the first collect with a missing method. Instrumenting the V2 runner
  is not done yet.
- **Tensor parallelism on 0.27 needs `VLLM_WORKER_MULTIPROC_METHOD=spawn`.** vLLM
  forks its workers there by default, and a forked process cannot re-initialize
  CUDA (`RuntimeError: Cannot re-initialize CUDA in forked subprocess`). This is
  vLLM's own setting and applies with or without nnsight.

**MoE reads differently on 0.27, and needs nothing from you.** That release
rebuilt the fused-experts layer around a factory and a modular kernel, and moved
the final all-reduce *inside* the layer — so its output is already the whole
value, and there is nothing for nnsight to gather. Measured on a two-rank
Qwen1.5-MoE: both ranks hand back the identical tensor. Through 0.26 the layer
left a per-rank partial and nnsight gathered it, as described above; either way
what you read is the whole thing.

The one case 0.27 still leaves partial is a layer built to defer its reduce
(`skip_final_all_reduce`), which nnsight gathers as before. A
sequence-parallel MoE layer is split by rows rather than summed — a different
correction, and one nnsight does not make yet, so that value is read as one rank's
rows.

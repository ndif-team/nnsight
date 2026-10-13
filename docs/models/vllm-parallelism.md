---
title: vLLM parallelism and architectures
one_liner: Tensor parallelism is transparent to a block; mixture-of-experts and hybrid trunks on vLLM.
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

`pipeline_parallel_size > 1` is a different split and is not supported. Each stage holds part of the trunk while the block is written against the whole tree and sent to every worker, so a stage that does not hold a module cannot resolve it: the engine builds, and every trace on it ends in `RuntimeError: UnknownPersistentIdError: Module:model.model.layers.12.self_attn`, naming a layer on the other stage. Shard with `tensor_parallel_size`.

## Mixture-of-experts models and expert parallelism

MoE models (Qwen-MoE, DeepSeek, Mixtral, ...) work with the same transparency, in both expert layouts vLLM offers on the same ranks:

- **default** (`enable_expert_parallel=False`): every rank holds a slice of every expert's matrices (the dense-MLP TP sharding, fused across experts);
- **expert parallel** (`enable_expert_parallel=True`): each rank holds `num_experts / world_size` whole experts.

The router (`mlp.gate`, a `ReplicatedLinear`) is full and identical on every rank, so reading or swapping router logits needs no gathering. The fused-experts module (`mlp.experts`, a `FusedMoE`) reads whole too: a layer that returns per-rank partial sums is all-reduced on access, and a write-back is divided by the group size so the model's own all-reduce reconstructs it exactly once (`tests/vllm/test_moe_batching.py`). The exception: a sequence-parallel MoE layer is split by rows rather than summed, which nnsight does not correct, so it reads as one rank's rows.

Individual experts are **not** addressable as submodules: vLLM stacks all local experts into fused weight tensors consumed by one grouped kernel, so there is no `experts[3]` at any parallelism level. The top-k selection and routing weights are computed inside that kernel too — recompute them from the logits. `mlp.gate.output` is `(logits, bias)`, `[tokens, num_experts]`; to ablate an expert, mask its router logit: `mlp.gate.output[0][:, e] = -inf`.

```python
with model.trace(prompt, temperature=0.0, max_tokens=6) as tracer:
    tops = list().save()
    for _ in tracer.iter[:6]:
        logits, _bias = model.model.layers[5].mlp.gate.output
        tops.append(logits[-1].topk(2).indices.clone())
```

On `TransformersModel` the same router-logit mask is a silent no-op; see
[ablating an MoE expert](../patterns/ablation.md#ablating-an-moe-expert).

## Hybrid (linear-attention) trunks

Qwen3-Next / Qwen3.5-style models interleave gated-delta-net layers with full-attention layers. Both are ordinary decoder-layer envoys; tell them apart by the child they carry, `layers[i].linear_attn` or `layers[i].self_attn`. The recurrent state lives in vLLM's state cache, not in any module output. `taps=` works on these models: a tapped engine pins `cudagraph_mode="FULL_DECODE_ONLY"` on any model vLLM reports as hybrid or attention-free (a full graph captured over a recurrent layer silently miscomputes the other batch composition), and tapped generation matches eager exactly. Checkpoints with a vision tower (`Qwen3_5ForConditionalGeneration`) load and trace on text; their decoder layers are at `model.language_model.model.layers`.

## vLLM versions

Tested versions and the model-runner pin: [vllm.md](vllm.md#vllm-versions). Scripts need an `if __name__ == "__main__":` guard on one card as much as under tensor parallelism; see [vllm.md](vllm.md#gotchas).

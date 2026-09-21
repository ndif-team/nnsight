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

## Pipeline parallelism

`pipeline_parallel_size > 1` splits the trunk by layers: each stage holds a contiguous run of layers, the first stage the embeddings, the last stage the final norm, the head and the sampler. The block is written against the whole tree and every stage runs all of it. What a stage does with a module it does not hold:

| use of a module on another stage | what happens |
|---|---|
| read `.output`, `.input`, `.inputs` | the block waits for the value; the owning stage sends it as its own copy of the block reads it |
| write `.output`, `.input`, `.skip()` | absorbed here, applied on the owning stage |
| `module.param("weight")` | the head's weight is on every stage from load; any other parameter is fetched from the owning stage when a request first reads it and dropped when that request finishes |
| `module.weight` (the attribute) | raises, naming the owning stage; use `param("weight")` |
| call the module, `model.model.norm(h)` | the module's state is fetched from the owning stage for the request and the call runs here on it; a module that computes from a buffer outside its state dict has to be called on its owner |
| `.source` of the module | fails to resolve any operation |

Saved values are collected from the last stage, whose copy of the block always runs to the end, and from the other stages for what only they hold. A read the block only saves (`h = layer.output.save()` with `h` not used again, or `kept.append(layer.output[0])` into a saved list the block only appends to) is not sent at all: the stage holding the module saves it there, and the saved value comes back from that stage at the end of the request. Every other read of another stage's module is sent as a copy taken as the block read it. A value saved on the stage that holds its module is a view of the engine's buffer, as on one GPU, so the "clone what you keep" rule of [vllm.md](vllm.md) applies to it and `NNSIGHT_VLLM_CLONE_READS` covers every stage alike. `tracer.iter` steps every stage together.

The ordering rule is the one a single GPU has: a read parks the block until the model reaches the location, so a location the forward passes meanwhile is out of order afterwards. On a later stage this means a read of a later stage's value parks the block until that stage runs, and the local layers passed in the meantime cannot be read after it. Read the local values first.

```python
model = VLLM("Qwen/Qwen2.5-14B-Instruct", pipeline_parallel_size=2)
with model.trace(prompt):
    early = model.model.layers[3].output[0]                 # stage 0's layer, local there
    normed = model.model.norm.output[0]                     # stage 1's, sent where needed
    logits = normed[-1] @ model.lm_head.param("weight").T   # the head, fetched to stage 0
    top1 = logits.argmax(-1).save()
```

## Mixture-of-experts models and expert parallelism

MoE models (Qwen-MoE, DeepSeek, Mixtral, ...) work with the same transparency, in both expert layouts vLLM offers on the same ranks:

- **default** (`enable_expert_parallel=False`): every rank holds a slice of every expert's matrices (the dense-MLP TP sharding, fused across experts);
- **expert parallel** (`enable_expert_parallel=True`): each rank holds `num_experts / world_size` whole experts.

The router (`mlp.gate`, a `ReplicatedLinear`) is full and identical on every rank, so reading router logits or swapping them (expert steering / expert-masking ablation) needs no gathering at all. The fused-experts module (`mlp.experts`, a `FusedMoE`) is the one MoE-specific case the batcher handles. A layer that returns **per-rank partial sums**, leaving the all-reduce to the block around it, is all-reduced into the true value on access, and a write-back is divided by the group size so that block's own all-reduce reconstructs the swapped value exactly once. Verified in `tests/vllm/test_moe_batching.py` against an HF reference. On 0.27 the fused layer usually reduces inside itself and hands back the whole value already, so there is nothing to gather; measured on a two-rank Qwen1.5-MoE, both ranks return the identical tensor. Either way what your block reads is the whole thing.

The case that still arrives partial is a layer built to defer its reduce (`skip_final_all_reduce`), which is gathered as above. A sequence-parallel MoE layer is split by rows rather than summed, which is a different correction and one nnsight does not make, so that value reads as one rank's rows.

Individual experts are **not** addressable as submodules: vLLM stacks all local experts into fused weight tensors consumed by one grouped kernel, so there is no `experts[3]` at any parallelism level. The top-k selection and routing weights are computed inside that kernel too — recompute them from the logits. `mlp.gate.output` is `(logits, bias)`, `[tokens, num_experts]`; to ablate an expert, mask its router logit: `mlp.gate.output[0][:, e] = -inf`.

```python
with model.trace(prompt, temperature=0.0, max_tokens=6) as tracer:
    tops = list().save()
    for _ in tracer.iter[:6]:
        logits, _bias = model.model.layers[5].mlp.gate.output
        tops.append(logits[-1].topk(2).indices.clone())
```

!!! warning "Masking the router logit is a vLLM recipe"
    On **`TransformersModel`** the same line is a silent no-op. A transformers MoE block
    calls its router as `_, top_k_weights, top_k_index = self.gate(hidden_states)`, and
    element `[0]`, the logits, is discarded, so masking it changes nothing: setting all 64
    logits to `-inf` gives `max|delta| = 0.0`. Edit the *selection* there instead, by
    writing the routing weights or expert indices in `mlp.gate.output[1]` / `[2]`.

    Two further traps on that side. With `norm_topk_prob=False`, masking a weight rescales
    every surviving expert, so a never-selected expert can show a larger effect than a real
    below-median one; ablate by selection, not by weight. And the router's `.input` /
    `.output` have no batch axis (`(B*T, E)`), so per-token statistics over a padded batch
    quietly include the pad rows.

## Hybrid (linear-attention) trunks

Qwen3-Next / Qwen3.5-style models interleave gated-delta-net layers with full-attention layers. Both are ordinary decoder-layer envoys; tell them apart by the child they carry, `layers[i].linear_attn` or `layers[i].self_attn`. The recurrent state lives in vLLM's state cache, not in any module output. `taps=` works on these models: a tapped engine pins `cudagraph_mode="FULL_DECODE_ONLY"` on any model vLLM reports as hybrid or attention-free (a full graph captured over a recurrent layer silently miscomputes the other batch composition), and tapped generation matches eager exactly. Checkpoints with a vision tower (`Qwen3_5ForConditionalGeneration`) load and trace on text; their decoder layers are at `model.language_model.model.layers`.

## vLLM versions

Everything on this page was run on **0.27.1**. The `vllm` extra carries no upper
bound, so `pip install "nnsight[vllm]"` takes whatever vLLM is current, and CI runs
`pytest tests/ --ignore=tests/vllm` — `tests/vllm` needs a GPU and is run by hand
against the release in the environment. Check the version you have rather than
assuming a range. Two things about the engine you build are worth knowing, because
nnsight arranges the first for you and the second is yours:

- **The model runner.** vLLM 0.27 ships a second GPU model runner and defaults to
  it for every non-MoE model. nnsight's interventions arrive by subclassing the
  original one, so it asks vLLM for that one (`VLLM_USE_V2_MODEL_RUNNER=0`) when it
  builds the engine. Setting that variable to `1` yourself is refused rather than
  overridden: the engine would otherwise come up with no interventions installed and
  fail at the first collect with a missing method. The V2 runner is not
  instrumented.
- **Workers are spawned, so a script needs an `if __name__ == "__main__":`
  guard.** Dispatching an engine initializes CUDA in your process, and vLLM then
  overrides `VLLM_WORKER_MULTIPROC_METHOD` to `spawn` itself, logging
  `We must use the spawn multiprocessing start method ... Reasons: CUDA is
  initialized`. Spawning re-imports the main module, so a `VLLM(...)` at module level
  builds a second engine in the child and the engine core dies with
  `An attempt has been made to start a new process before the current process has
  finished its bootstrapping phase`. This is unconditional: one card and `mode="sync"`
  hit it as readily as `tensor_parallel_size=8`. Notebooks have no main module to
  re-import and are unaffected. See the gotcha on [the engine page](vllm.md).

# NNsight vLLM Intervention Guide

This guide mirrors the [main nnsight documentation](../../../../CLAUDE.md) (written for HuggingFace Transformers) and documents how each pattern translates to the vLLM backend — what works identically, what differs, and what is blocked.

For each feature, the structure is:
1. **HF pattern** — what users expect (from the main docs)
2. **vLLM gap** — how vLLM's internals differ
3. **Status** — works as-is, requires different syntax, or hard-blocked

---

## Hard Blocks

These features are **fundamentally impossible** on vLLM due to C/CUDA-level architecture. Use the HF backend if your workflow requires them.

| Feature | Why blocked | Impact |
|---------|------------|--------|
| **Gradients / backward** | `torch.inference_mode()` globally disables gradients. `requires_grad_(True)` raises `RuntimeError`. | Integrated gradients, saliency maps, GradCAM, all gradient-based attribution. |
| **Attention weights** | PagedAttention computes attention in C/CUDA. No Python-level weight tensor exists. | Attention visualization, head pruning, induction head detection. |
| **Source tracing (fused modules)** | Fused kernels have trivial Python wrappers. `.source` shows only the delegate call. Non-fused modules (e.g. `self_attn`) work fine. | Fine-grained intervention inside fused ops (layernorm, act_fn). |

---

## Remaining Differences

These require different vLLM syntax:

| Difference | HF | vLLM | Notes |
|---|---|---|---|
| **Decoder output** | `(combined_hidden,)` 1-tuple | `(mlp_output, residual)` 2-tuple | Combine manually: `out[0] + out[1]` |
| **Decoder input** | `.input` → float hidden states | `.input` → int64 positions | Use `.inputs[0][1]` for hidden states |
| **LayerNorm output** | Single tensor | `(normalized, residual)` tuple | Use `.output[0]` |
| **RowParallel output** | Single tensor | `(output, bias)` tuple | Use `.output[0]` for down\_proj, o\_proj |
| **Logits** | `model.lm_head.output` `[1, seq, vocab]` | `model.logits.output` `[1, vocab]` (last token only) | Different module path and scope |
| **Generation** | `model.generate(max_new_tokens=N)` | `model.trace(max_tokens=N)` | Different method and kwarg |
| **gate/up proj** | `mlp.gate_proj`, `mlp.up_proj` (separate) | `mlp.gate_up_proj` (merged) | `gate, up = output.chunk(2, dim=-1)` |
| **q/k/v proj** | `attn.q_proj`, `attn.k_proj`, `attn.v_proj` | `attn.qkv_proj` (merged) | `q, k, v = output.split([q, kv, kv], dim=-1)` |
| **Module skip** | `layer.skip(value)` | Must provide `(hidden, residual)` tuple | `layer.skip((zeros, value))` |
| **`__name__` guard** | Not required | `if __name__ == "__main__":` required | vLLM uses `spawn` multiprocessing |
| **Module calls outside trace** | Works (weights are local) | Fails (weights on meta device) | Must call modules inside trace |

---

## Prefix Caching

NNsight **disables prefix caching by default** (`enable_prefix_caching=False`). The impact of prefix caching on interventions is not fully understood. We disable it as a precaution.

This is also why NNsight does not support [SGLang](https://github.com/sgl-project/sglang) — SGLang does not allow disabling its RadixAttention prefix cache.

---

## Setup

### HF
```python
from nnsight import LanguageModel
model = LanguageModel("Qwen/Qwen2-0.5B", device_map="auto", dispatch=True)
```

### vLLM
```python
from nnsight.modeling.vllm import VLLM

if __name__ == "__main__":
    model = VLLM("Qwen/Qwen2-0.5B", tensor_parallel_size=1,
                 gpu_memory_utilization=0.3, dispatch=True)
```

**`__name__` guard is required.** vLLM uses `spawn` multiprocessing to create engine worker processes. Without the guard, the spawned process re-imports your script and re-executes all module-level code, causing infinite recursion or duplicate engine creation.

---

## Accessing Hidden States

### HF pattern
```python
with model.trace("Hello"):
    hidden = model.model.layers[-1].output[0].save()
# shape [batch, seq, hidden_dim]
```

### vLLM gap
Decoder layers return `(mlp_output, residual)` — two separate tensors instead of a combined hidden state.

### Status: **Different format — combine manually**
```python
with model.trace("Hello"):
    # output is (mlp_output, residual) — combine for HF-equivalent hidden state
    combined = (model.model.layers[-1].output[0] + model.model.layers[-1].output[1]).save()
# shape [tokens, hidden_dim] — 2D instead of HF's 3D
```

---

## Saving Values

### HF pattern
```python
with model.trace("Hello"):
    hidden = model.model.layers[0].output[0].save()
# hidden is a real tensor after the context exits
```

### vLLM gap
vLLM's `fused_add_rms_norm` mutates tensors in-place after hooks fire, silently corrupting `.save()`'d references.

### Status: **Mitigated — general inference-mode cloning**
```python
with model.trace("Hello"):
    hidden = model.model.layers[0].output[0].save()
# Mutation-safe — nnsight auto-clones inference-mode tensors
```

---

## Accessing Logits

### HF pattern
```python
with model.trace("Hello"):
    logits = model.lm_head.output.save()   # all tokens, shape [batch, seq, vocab]
```

### vLLM gap
vLLM only computes `lm_head` on the last token (sufficient for sampling). Logits are exposed through `model.logits`, not `model.lm_head`.

### Status: **Different syntax required**
```python
with model.trace("Hello"):
    logits = model.logits.output.save()   # last token only, shape [1, vocab]
```

---

## Accessing Layer Inputs

### HF pattern
```python
with model.trace("Hello"):
    layer_input = model.model.layers[0].input.save()
# Returns float hidden states
```

### vLLM gap
Decoder layer forward signature is `(positions, hidden_states, residual, ...)`. Raw `.input` returns `positions` (int64), not the hidden state.

### Status: **Different format — use `.inputs`**
```python
with model.trace("Hello"):
    # .input returns positions (int64). Use .inputs for full args:
    args, kwargs = model.model.layers[4].inputs
    hidden_states = args[1]     # float tensor
    residual = args[2]          # float tensor (None for layer 0)
    combined = hidden_states + residual if residual is not None else hidden_states
    combined = combined.save()
```

---

## Modifying Activations — Ablation

### HF pattern
```python
with model.trace("Hello"):
    model.model.layers[0].output[0][:] = 0
```

### vLLM gap
In-place assignment on the raw dual-stream output would only zero one stream.

### Status: **Different format — zero both streams**
```python
with model.trace("Hello"):
    # Must zero both streams for full ablation
    model.model.layers[0].output[0][:] = 0  # mlp_output stream
    model.model.layers[0].output[1][:] = 0  # residual stream
```

---

## Modifying Activations — Steering

### HF pattern
```python
steering_vector = torch.randn(hidden_dim)
with model.trace("Hello"):
    model.model.layers[10].output[0][:, -1, :] += steering_vector
```

### vLLM gap
Same dual-stream issue as ablation, plus 2D tensor layout.

### Status: **Different format — steer the appropriate stream**
```python
steering_vector = torch.randn(hidden_dim, dtype=torch.bfloat16).cuda()
with model.trace("Hello"):
    # Add to mlp_output stream (output[0]) or residual stream (output[1])
    model.model.layers[10].output[0][-1, :] += steering_vector
```

---

## Activation Patching

### HF pattern
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.model.layers[5].output[0][-1, :]
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.model.layers[5].output[0][-1, :] = clean_hs
        logits = model.lm_head.output.save()
```

### vLLM gap
vLLM's internal batching does not guarantee execution order across invokes. Each invoke's request may be scheduled in any order by the engine. This means **barriers and cross-invoke dependencies are not supported** — there is no way to ensure invoke 1's output is available before invoke 2 runs.

Multiple invokes in a single trace are fine for independent operations (e.g., running the same intervention on different prompts), but they must not reference each other's values.

Also note: `tracer.stop()` in vLLM only stops the invoke that called it. Other invokes in the same trace continue running independently.

### Status: **Different syntax required** — use two separate traces for dependencies
```python
# Step 1: extract hidden states from source prompt
with model.trace("The Eiffel Tower is in"):
    clean_hs = model.model.layers[5].output[0][-1, :].save()

# Step 2: patch into target prompt (separate trace — guarantees ordering)
with model.trace("The Colosseum is in"):
    model.model.layers[5].output[0][-1, :] = clean_hs.to(model.model.layers[5].output[0].device)
    logits = model.logits.output.save()
```

---

## Logit Lens

### HF pattern
```python
with model.trace("The Eiffel Tower is in"):
    for i in range(num_layers):
        hs = model.model.layers[i].output[0]
        logits = model.lm_head(model.model.norm(hs))
        print(model.tokenizer.decode(logits.argmax(dim=-1)[0][-1]))
```

### vLLM gap
Dual-stream outputs must be combined manually. Fused norm returns a tuple. Additionally, **module calls must happen inside the trace** — unlike HF where you can call `model.lm_head(hs)` on saved tensors outside a trace, vLLM's model weights live in a separate worker process and are not accessible from the client.

### Status: **Different format — combine streams manually**
```python
with model.trace("The Eiffel Tower is in"):
    for i in range(num_layers):
        hs = model.model.layers[i].output[0] + model.model.layers[i].output[1]  # combine dual streams
        normed = model.model.norm(hs)
        if isinstance(normed, tuple):   # fused norm returns (normalized, residual)
            normed = normed[0]
        logits = model.lm_head(normed)
        print(model.tokenizer.decode(logits.argmax(dim=-1)[-1]))
```

---

## Multi-Token Generation

### HF pattern
```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()
    for step in tracer.iter[:]:
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))
    output = model.generator.output.save()
```

### vLLM gap
Different generation API, different logits module.

### Status: **Different syntax required**
```python
import nnsight
with model.trace("Hello", max_tokens=5) as tracer:
    logits = nnsight.save(list())
    for step in tracer.iter[:]:
        logits.append(model.logits.output)
```

- `model.trace(max_tokens=N)` instead of `model.generate(max_new_tokens=N)`
- `model.logits.output` instead of `model.lm_head.output`
- `model.samples.output` provides the sampled token

---

## Module Skip

### HF pattern
```python
with model.trace("Hello"):
    layer0_out = model.model.layers[0].output
    model.model.layers[1].skip(layer0_out)
```

### vLLM gap
`skip()` provides an HF-shaped value, but vLLM's fused norm expects `(hidden_states, residual)` pairs. Raw `skip()` crashes the engine.

### Status: **Different format — must provide (hidden, residual) tuple**
```python
with model.trace("Hello"):
    layer0_out = model.model.layers[0].output  # already (mlp_output, residual)
    model.model.layers[1].skip(layer0_out)     # pass the raw tuple through
```

Skip values must match vLLM's expected `(hidden_states, residual)` format. Passing a single tensor will crash the engine (deferred exception prevents engine death but the skip still fails).

---

## Early Stop (tracer.stop())

### HF pattern
```python
with model.trace("Hello") as tracer:
    hs = model.model.layers[0].output[0].save()
    tracer.stop()
    # Code after stop() does not execute
```

### vLLM gap
Exceptions (including `EarlyStopException` from `stop()`) were re-raised inside PyTorch hooks during the forward pass, unwinding vLLM's `execute_model()` and permanently killing the engine.

### Status: **Solved — deferred exception handling**
```python
with model.trace("Hello") as tracer:
    hs = model.model.layers[0].output[0].save()
    tracer.stop()
    # Code after stop() does not execute
# hs is a real tensor; engine survives
```

Exceptions are deferred (stored, not raised) during the forward pass. The forward completes normally, then exceptions are surfaced at the trace boundary on the client side. `EarlyStopException` is filtered as intentional control flow. User errors (e.g. `IndexError`) are re-raised with the original type and traceback.

**Note:** In vLLM, `tracer.stop()` only stops the invoke that called it. Other invokes in the same trace continue running independently (see [Activation Patching](#activation-patching) for details on invoke independence).

---

## Error Handling

### HF pattern
```python
try:
    with model.trace("Hello"):
        bad = model.model.layers[100].output[0].save()  # IndexError
except IndexError as e:
    print(e)  # Error raised, model still usable
```

### vLLM gap
Same issue as `tracer.stop()` — exceptions inside hooks killed the engine.

### Status: **Solved — deferred exception handling**
```python
try:
    with model.trace("Hello"):
        bad = model.model.layers[100].output[0].save()
except IndexError as e:
    print(e)  # Error raised at trace boundary, engine survives

# Engine is still alive — subsequent traces work
with model.trace("Hello"):
    logits = model.logits.output.save()
```

Per-mediator exception isolation ensures one failing trace doesn't affect other concurrent traces in the same batch. Each trace's error is reported independently.

---

## LayerNorm Output

### HF pattern
```python
with model.trace("Hello"):
    normed = model.model.layers[0].input_layernorm.output   # single tensor
```

### vLLM gap
Fused RMSNorm returns `(normalized, residual)` tuple instead of a single tensor.

### Status: **Different format — extract from tuple**
```python
with model.trace("Hello"):
    normed = model.model.layers[0].input_layernorm.output[0]   # extract normalized tensor
    # output[1] is the residual
```

**Note:** LayerNorm *inputs* also differ (2 args vs 1 in HF).

---

## Linear Layer Outputs (down_proj, o_proj)

### HF pattern
```python
with model.trace("Hello"):
    out = model.model.layers[0].mlp.down_proj.output   # single tensor
```

### vLLM gap
`RowParallelLinear` returns `(output, bias)` 2-tuple.

### Status: **Different format — extract from tuple**
```python
with model.trace("Hello"):
    out = model.model.layers[0].mlp.down_proj.output[0]   # extract tensor from (output, bias) tuple
```

---

## Merged Projections (gate_up_proj, qkv_proj)

### HF pattern
```python
with model.trace("Hello"):
    gate = model.model.layers[0].mlp.gate_proj.output
    up   = model.model.layers[0].mlp.up_proj.output
```

### vLLM gap
vLLM merges `gate_proj` + `up_proj` into `gate_up_proj` and `q/k/v_proj` into `qkv_proj` for memory bandwidth.

### Status: **Different syntax required**
```python
with model.trace("Hello"):
    merged = model.model.layers[0].mlp.gate_up_proj.output
    gate, up = merged.chunk(2, dim=-1)

    merged_qkv = model.model.layers[0].self_attn.qkv_proj.output
    q, k, v = merged_qkv.split([q_size, kv_size, kv_size], dim=-1)
```

---

## Attention Weights

### HF pattern
```python
with model.trace("Hello"):
    weights = model.model.layers[0].self_attn.source.attention_interface_0.output[0]
```

### Status: **Hard block — not available**

PagedAttention computes attention entirely in C/CUDA. No Python-level weight tensor exists. No workaround.

---

## Gradients

### HF pattern
```python
with model.trace("Hello"):
    hs = model.model.layers[-1].output[0]
    hs.requires_grad_(True)
    with model.lm_head.output.sum().backward():
        grad = hs.grad.save()
```

### Status: **Hard block — not available**

`torch.inference_mode()` globally disables gradients. `requires_grad_(True)` raises `RuntimeError`. No workaround.

---

## Source Tracing

### HF pattern
```python
print(model.model.layers[0].self_attn.source)   # shows all operations
with model.trace("Hello"):
    sdpa = model.model.layers[0].self_attn.source.attention_interface_0.output
```

### vLLM gap
Fused modules have trivial Python wrappers — `.source` shows only the delegate call.

### Status: **Partially works**

Non-fused modules (e.g. `self_attn`) work fine — `.source` reveals `qkv_proj`, `split`, `rotary_emb`, `attn`, `o_proj`. Fused modules (`input_layernorm`, `act_fn`) show only 1 operation.

---

## Caching

### HF pattern
```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(modules=[model.model.layers[0]])
hidden = cache['model.model.layers.0'].output[0]
```

### Status: **Works — values are in raw vLLM format**
```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(modules=[model.model.layers[0]])
# Decoder layer output is (mlp_output, residual) tuple
hidden = cache.model.layers[0].output[0] + cache.model.layers[0].output[1]  # combine manually
```

Cached values are auto-cloned (mutation-safe) but in raw vLLM format.

---

## Tensor Parallelism

### HF pattern
```python
# HF with device_map="auto" distributes layers across GPUs.
# Each layer's output is the full unsharded tensor.
model = LanguageModel("model", device_map="auto", dispatch=True)
```

### vLLM gap
vLLM shards `ColumnParallelLinear` and `RowParallelLinear` across GPUs. Raw sub-module outputs are per-GPU shards, not full tensors. Worker threads default to CUDA stream 0, which races with vLLM's compute stream under TP.

### Status: **Layer-level solved, sub-module access crashes**

```python
model = VLLM("Qwen/Qwen2-0.5B", tensor_parallel_size=2,
             gpu_memory_utilization=0.3, dispatch=True)
```

**Layer-level interventions work identically to tp=1.** `VLLMBatcher` transparently gathers sharded tensors (via `tensor_model_parallel_all_gather` / `all_reduce`) so users see full unsharded values, then splits them back before returning to vLLM. CUDA stream propagation ensures worker threads use the correct compute stream, eliminating non-determinism.

**Sub-module access (qkv_proj, gate_up_proj, etc.) crashes the engine with tp≥2.** This is unrecoverable — the vLLM process must be restarted.

Recommendations:
1. Use layer-level interventions only with tp≥2
2. Avoid sub-module access — crashes the engine
3. Test with tp=1 first, then verify at the layer level with tp=2

---

## Quick Reference

| Operation | HF | vLLM | Notes |
|-----------|-----|------|-------|
| **Save hidden** | `layer.output[0].save()` | `(layer.output[0] + layer.output[1]).save()` | Combine dual streams |
| **Layer input** | `layer.input` | `.inputs[0][1]` for hidden states | `.input` returns positions |
| **Ablation** | `output[0][:] = 0` | Zero both `output[0]` and `output[1]` | Dual stream |
| **Steer** | `output[0][-1, :] += v` | `output[0][-1, :] += v` | Steers mlp stream only |
| **Norm output** | tensor | `.output[0]` | Extract from tuple |
| **Linear output** | tensor | `.output[0]` | Extract from `(output, bias)` |
| **Module skip** | `layer.skip(v)` | `layer.skip((h, r))` | Must provide tuple |
| **tracer.stop()** | Works | Works | Deferred exception |
| **Error handling** | Error raised | Error raised | Deferred, engine survives |
| **Logits** | `model.lm_head.output` | `model.logits.output` | Different path, last-token only |
| **Generation** | `model.generate(max_new_tokens=N)` | `model.trace(max_tokens=N)` | Different API |
| **gate/up proj** | Separate modules | `gate_up_proj` merged | `.chunk(2)` |
| **q/k/v proj** | Separate modules | `qkv_proj` merged | `.split()` |
| **Attn weights** | Available | **Blocked** | PagedAttention (C/CUDA) |
| **Gradients** | Available | **Blocked** | `inference_mode` |
| **Source tracing** | All modules | Python wrappers only | Fused modules blocked |

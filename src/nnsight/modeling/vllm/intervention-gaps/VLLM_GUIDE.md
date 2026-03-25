# NNsight vLLM Intervention Guide

This guide mirrors the [main nnsight documentation](../../../../CLAUDE.md) (written for HuggingFace Transformers) and shows how each pattern translates to the vLLM backend. For every example, we show:

1. **HF pattern** — the doc example as-written
2. **vLLM equivalent** — the working vLLM version
3. **Why it differs** — root cause and gap reference

All examples tested on **Qwen2.5-0.5B** with vLLM 0.15.1, nnsight on branch `vllm-intervention-gaps`.

> **Key architectural difference:** vLLM uses a **dual residual stream**. Each decoder layer returns `(mlp_output, residual)` as separate tensors. The next layer's `fused_add_rms_norm` computes `RMSNorm(mlp_output + residual)`. Only the **sum** of both streams equals the HF-equivalent hidden state.

---

## Table of Contents

1. [Setup](#setup)
2. [Accessing Hidden States](#accessing-hidden-states)
3. [Accessing Logits](#accessing-logits)
4. [Accessing Layer Inputs](#accessing-layer-inputs)
5. [Saving Values](#saving-values)
6. [Modifying Activations — Ablation](#modifying-activations--ablation)
7. [Modifying Activations — Steering](#modifying-activations--steering)
8. [Activation Patching](#activation-patching)
9. [Logit Lens](#logit-lens)
10. [Multi-Token Generation](#multi-token-generation)
11. [LayerNorm Output](#layernorm-output)
12. [Module Architecture Differences](#module-architecture-differences)
13. [Attention Weights](#attention-weights)
14. [Gradients](#gradients)
15. [Module Skip](#module-skip)
16. [Source Tracing](#source-tracing)
17. [Caching](#caching)
18. [Tensor Parallelism (tp=2)](#tensor-parallelism-tp2)
19. [Quick Reference Table](#quick-reference-table)

---

## Setup

### HF (LanguageModel)
```python
from nnsight import LanguageModel
model = LanguageModel("Qwen/Qwen2.5-0.5B", device_map="auto", dispatch=True)
```

### vLLM
```python
from nnsight.modeling.vllm import VLLM
model = VLLM("Qwen/Qwen2.5-0.5B", tensor_parallel_size=1,
             gpu_memory_utilization=0.3, dtype=torch.float16, dispatch=True)
```

**Note:** vLLM traces should include sampling parameters for deterministic results:
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    ...
```

---

## Accessing Hidden States

### HF pattern
```python
with model.trace("Hello"):
    hidden = model.transformer.h[-1].output[0].save()
# hidden is the full residual-stream hidden state, shape [batch, seq, hidden_dim]
```

### vLLM equivalent
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    hidden = (model.model.layers[-1].output[0].clone()
              + model.model.layers[-1].output[1].clone()).save()
# hidden is the combined hidden state, shape [total_tokens, hidden_dim]
```

### Why it differs

**Three differences compound here:**

1. **Gap 1.1 — In-place mutation:** vLLM's `fused_add_rms_norm` mutates tensors after hooks fire. `.save()` stores a reference to the mutated tensor. **Must use `.clone().save()`**.

2. **Gap 1.2 — Dual residual stream:** vLLM decoder layers return `(mlp_output, residual)` as separate tensors. `output[0]` alone is just the MLP component — NOT the full hidden state. **Must add both: `output[0] + output[1]`**.

3. **Gap 3.1 — Flat 2D layout:** vLLM uses continuous batching with shape `[total_tokens, hidden_dim]` instead of HF's `[batch, seq_len, hidden_dim]`. Indexing like `[:, -1, :]` will fail.

**Verified:** `output[0].clone() + output[1].clone()` matches HF hidden state within fp16 variance (max_diff ≈ 0.06).

---

## Accessing Logits

### HF pattern
```python
with model.trace("Hello"):
    logits = model.lm_head.output.save()       # shape [batch, seq, vocab]
    last_logits = model.lm_head.output[:, -1]   # last token logits
```

### vLLM equivalent
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    logits = model.logits.output.save()   # shape [1, vocab_size]
```

### Why it differs

vLLM provides a special `model.logits` module that captures the final logit output. `model.lm_head.output` may also work (it did in our tests with Qwen2.5-0.5B), but `model.logits.output` is the documented vLLM-specific pattern and is more reliable across models.

**Note:** `model.logits.output` shape is `[1, vocab_size]` (already last-token only in single-token mode), not `[batch, seq, vocab]`.

---

## Accessing Layer Inputs

### HF pattern
```python
with model.trace("Hello"):
    layer_input = model.transformer.h[0].input.save()
    # Returns float hidden states, shape [batch, seq, hidden_dim]
```

### vLLM equivalent
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    args, kwargs = model.model.layers[4].inputs
    hidden = (args[1].clone() + args[2].clone()).save()
    # args[0] = positions (int64), args[1] = hidden_states, args[2] = residual
```

### Why it differs

**Gap 1.3:** vLLM decoder layers have signature `forward(positions, hidden_states, residual, ...)`. The `.input` property returns the first positional argument, which is **int64 position IDs** — not float hidden states.

Use `.inputs` (plural) to get all arguments, then pick `args[1]` (hidden_states) and `args[2]` (residual). Add them to reconstruct the HF-equivalent input.

**Verified:** `.input` returns dtype=int64 with values like `[0, 1, 2, 3, 4]`. `args[1]+args[2]` matches HF within fp16 variance (max_diff ≈ 0.06).

---

## Saving Values

### HF pattern
```python
with model.trace("Hello"):
    hidden = model.transformer.h[0].output[0].save()
```

### vLLM — ALWAYS use `.clone().save()`
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    hidden = model.model.layers[0].output[0].clone().save()
```

### Why it differs

**Gap 1.1:** vLLM's fused CUDA kernels (`fused_add_rms_norm`) mutate activation tensors **in-place after** nnsight's hooks capture them. `.save()` stores a reference, so the saved value silently becomes the post-mutation value.

**Verified:** `.save()` vs `.clone().save()` on layer 5 output[0] shows max_diff=18.95 — the saved value is corrupted without `.clone()`.

---

## Modifying Activations — Ablation

### HF pattern
```python
with model.trace("Hello"):
    model.transformer.h[0].output[0][:] = 0    # Zero the hidden state
```

### vLLM equivalent
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    model.model.layers[0].output = (
        torch.zeros_like(model.model.layers[0].output[0]),
        torch.zeros_like(model.model.layers[0].output[1]),
    )
```

### Why it differs

Two issues:

1. **Gap 1.2:** `output[0][:] = 0` only zeros the MLP stream. The residual stream (`output[1]`) leaks through to the next layer. For HF-equivalent ablation, **must zero both streams**.

2. **Tuple assignment:** `layer.output[0] = x` raises `TypeError: 'tuple' object does not support item assignment`. Must replace the entire tuple: `layer.output = (new0, new1)`.

---

## Modifying Activations — Steering

### HF pattern
```python
steering_vector = torch.randn(768)  # Pre-computed direction

with model.trace("Hello"):
    model.transformer.h[10].output[0][:, -1, :] += steering_vector * 0.5
```

### vLLM equivalent
```python
steering_vector = torch.randn(896, dtype=torch.float16).cuda() * 0.5

with model.trace("Hello", temperature=0.0, top_p=1):
    out0 = model.model.layers[10].output[0].clone()
    out1 = model.model.layers[10].output[1].clone()
    out0[-1, :] += steering_vector  # 2D indexing: [tokens, hidden]
    model.model.layers[10].output = (out0, out1)
```

### Why it differs

1. **Gap 3.1:** `[:, -1, :]` (3D) fails because vLLM tensors are 2D `[tokens, hidden]`. Use `[-1, :]` instead.

2. **Tuple replacement:** Can't modify `output[0]` in-place on the tuple. Must clone both, modify, then replace the whole tuple.

3. **Steering is stream-agnostic:** Adding `v` to **either** `output[0]` or `output[1]` produces identical results. The next layer's `fused_add_rms_norm` sums them, so `(out0+v) + out1 = out0 + (out1+v)`. Adding to `output[0]` is conventional.

**Verified:** `steer_A` (add to output[0]) vs `steer_B` (add to output[1]): max_diff=0.004, cosine=1.000.

---

## Activation Patching

### HF pattern
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :]
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[0][:, -1, :] = clean_hs
        logits = model.lm_head.output.save()
```

### vLLM equivalent
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in", temperature=0.0, top_p=1):
        # Get combined hidden state at last token
        clean_hs = (model.model.layers[5].output[0].clone()[-1, :]
                    + model.model.layers[5].output[1].clone()[-1, :]).save()
        barrier()
    with tracer.invoke("The Colosseum is in", temperature=0.0, top_p=1):
        barrier()
        # Patch: adjust mlp stream so out0 + out1 = target
        out0 = model.model.layers[5].output[0].clone()
        out1 = model.model.layers[5].output[1].clone()
        out0[-1, :] = clean_hs - out1[-1, :]
        model.model.layers[5].output = (out0, out1)
        logits = model.logits.output.save()
```

### Why it differs

Combines all previous gaps:
- **2D indexing** (`[-1, :]` not `[:, -1, :]`)
- **Dual residual stream** (must combine `output[0]+output[1]` for extraction, and decompose back for injection)
- **Clone** to avoid mutation corruption
- **Tuple replacement** (can't assign to tuple elements)

**Simpler patching alternative** — if you don't need to preserve the decomposition:
```python
# Put entire target in one stream, zero the other
model.model.layers[5].output = (clean_hs_full, torch.zeros_like(out1))
```
All three decompositions `(target, zeros)`, `(target-res, res)`, `(zeros, target)` produce bit-identical results.

---

## Logit Lens

### HF pattern
```python
with model.trace("The Eiffel Tower is in"):
    for i in range(12):
        hs = model.transformer.h[i].output[0]
        logits = model.lm_head(model.transformer.ln_f(hs))
        tokens = logits.argmax(dim=-1).save()
        print(f"Layer {i}:", model.tokenizer.decode(tokens[0][-1]))
```

### vLLM equivalent
```python
with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
    for i in range(24):
        # Combine dual residual streams
        hs = (model.model.layers[i].output[0].clone()
              + model.model.layers[i].output[1].clone())
        # Apply final norm (may return tuple on vLLM)
        normed = model.model.norm(hs)
        if isinstance(normed, tuple):
            normed = normed[0]
        logits = model.lm_head(normed)
        tokens = logits.argmax(dim=-1).save()
        print(f"Layer {i}:", model.tokenizer.decode(tokens[-1]))
```

### Why it differs

1. **Gap 1.2:** Must combine `output[0] + output[1]` to get the full hidden state
2. **Gap 1.4:** `model.model.norm(hs)` may return a tuple `(normalized, residual)` when the fused kernel path is taken. Use `normed[0]` if it's a tuple.
3. **2D layout:** Use `tokens[-1]` not `tokens[0][-1]`

**Verified:** Logit lens produces coherent predictions (e.g., "Paris" at deeper layers for "The Eiffel Tower is in").

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

### vLLM equivalent
```python
import nnsight

with model.trace("Hello", max_tokens=3, temperature=0.0, top_p=1) as tracer:
    logits = nnsight.save(list())
    for step in tracer.iter[:]:
        logits.append(model.logits.output)
```

### Why it differs

- vLLM uses `model.trace(max_tokens=N)` instead of `model.generate(max_new_tokens=N)`
- Use `model.logits.output` instead of `model.lm_head.output`
- Use `nnsight.save(list())` instead of `list().save()` for more reliable behavior
- vLLM also provides `model.samples.output` for the sampled token

---

## LayerNorm Output

### HF pattern
```python
with model.trace("Hello"):
    normed = model.transformer.ln_f(hidden_states)  # Returns single tensor
```

### vLLM behavior
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    norm_out = model.model.layers[0].input_layernorm.output
    # Returns 2-tuple: (normalized, new_residual)
    normed = norm_out[0]      # The actual normalized value
    residual = norm_out[1]    # Updated residual stream
```

### Why it differs

**Gap 1.4:** vLLM's `fused_add_rms_norm` computes `RMSNorm(x + residual)` and returns both the normalized value and the updated residual as a tuple. HF's `RMSNorm` just returns the normalized tensor.

**Gap 1.5:** The input also differs — vLLM's layernorm takes 2 args `(x, residual)`, HF's takes 1 `(hidden_states)`.

---

## Module Architecture Differences

### MLP: gate_proj/up_proj merged

**HF:** Separate `gate_proj`, `up_proj`, `down_proj`
```python
model.transformer.h[0].mlp.gate_proj.output  # Works
model.transformer.h[0].mlp.up_proj.output    # Works
```

**vLLM:** Merged into single `gate_up_proj`
```python
model.model.layers[0].mlp.gate_up_proj.output  # Single merged output
# To split: gate, up = output.chunk(2, dim=-1)
```

**Gap 2.1:** vLLM uses `MergedColumnParallelLinear` for memory bandwidth efficiency.

### Attention: q/k/v merged

**HF:** Separate `q_proj`, `k_proj`, `v_proj`
```python
model.transformer.h[0].self_attn.q_proj.output  # Works
```

**vLLM:** Merged into single `qkv_proj`
```python
model.model.layers[0].self_attn.qkv_proj.output  # Single merged output
# To split: q, k, v = output.split([q_size, kv_size, kv_size], dim=-1)
```

**Gap 2.2:** vLLM uses `QKVParallelLinear` for tensor parallelism support.

### Linear outputs return tuples

**HF:** `down_proj.output` and `o_proj.output` return single tensors.

**vLLM:** They return `(output, bias)` 2-tuples.
```python
# Use output[0] for the actual tensor
actual_output = model.model.layers[0].mlp.down_proj.output[0]
```

**Gap 2.3:** vLLM uses `RowParallelLinear` which returns `(output, output_bias)`.

---

## Attention Weights

### HF pattern
```python
with model.trace("Hello"):
    attn_weights = model.transformer.h[0].attn.source.attention_interface_0.output[0]
```

### vLLM: NOT AVAILABLE

**Gap 3.2:** vLLM's PagedAttention computes attention entirely in C/CUDA. No Python-level attention weight tensor exists. `self_attn.output` returns a single tensor (the attention output), not a tuple with weights.

**No workaround.** This is a fundamental architectural limitation of PagedAttention.

---

## Gradients

### HF pattern
```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    logits = model.lm_head.output
    with logits.sum().backward():
        grad = hs.grad.save()
```

### vLLM: NOT AVAILABLE

**Gap 4.1:** vLLM wraps all execution in `torch.inference_mode()`, which globally disables gradient tracking. `requires_grad_(True)` raises `RuntimeError`.

**No workaround.** Integrated gradients, saliency maps, GradCAM, and all gradient-based attribution methods are blocked.

---

## Module Skip

### HF pattern
```python
with model.trace("Hello"):
    layer0_out = model.transformer.h[0].output
    model.transformer.h[1].skip(layer0_out)
```

### vLLM: BROKEN

**Gap 4.3:** `skip()` with a single tensor (or even a tuple) fails because the next layer's `fused_add_rms_norm` expects a specific `(hidden_states, residual)` pair structure. The engine crashes with `EngineDeadError`.

**No clean workaround.** Would require manually constructing the correct dual-stream tuple that the fused norm expects.

---

## Source Tracing

### HF pattern
```python
# View all operations in attention forward
print(model.transformer.h[0].attn.source)

# Access specific operation
with model.trace("Hello"):
    sdpa = model.transformer.h[0].attn.source.attention_interface_0.output
```

### vLLM — works on Python wrappers, limited on fused modules
```python
# self_attn is a Python wrapper — source tracing works well
print(model.model.layers[0].self_attn.source)
# Shows: qkv_proj, split, rotary_emb, attn, o_proj (5+ operations)

# input_layernorm is fused — only shows delegate call
print(model.model.layers[0].input_layernorm.source)
# Shows: 1 operation (the CUDA delegate)
```

### Why it differs

**Gap 4.2:** Fused modules (`input_layernorm`, `act_fn`, PagedAttention) have trivial Python wrappers that delegate to C/CUDA. `.source` can only show Python-level operations, so fused modules show just 1 delegate call.

**Non-fused modules work fine.** `self_attn.source` reveals `qkv_proj`, `split`, `rotary_emb`, `attn`, `o_proj` — all hookable.

---

## Caching

### HF pattern
```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(modules=[model.transformer.h[0]])

out = cache['model.transformer.h.0'].output
```

### vLLM equivalent
```python
with model.trace("Hello", temperature=0.0, top_p=1) as tracer:
    cache = tracer.cache(modules=[model.model.layers[0]])

out = cache.model.layers[0].output
# Note: out is a tuple (mlp_output, residual) — vLLM semantics
# Combined hidden state: out[0] + out[1]
```

Caching works on vLLM, but cached values reflect vLLM's semantics (dual residual stream, potential in-place mutation).

---

## Tensor Parallelism (tp=2)

With `tensor_parallel_size=2`, vLLM shards certain modules across GPUs. This introduces additional differences from tp=1.

### Verified: tp=2 on Qwen2.5-0.5B (GPUs 0,7)

| Test | tp=1 | tp=2 | Status |
|------|------|------|--------|
| `layer.output[0]` shape | `[2, 896]` | `[2, 896]` | **Same** — all-reduced to full hidden_dim |
| `layer.output[1]` shape | `[2, 896]` | `[2, 896]` | **Same** — residual is replicated |
| `logits.output` shape | `[1, 151936]` | `[1, 151936]` | **Same** — full vocab |
| Sub-module access (qkv_proj, gate_up_proj, etc.) | Works | **Engine crash** | **New tp=2 gap** |

### Key findings

**Layer-level interventions work identically to tp=1.** The layer output tensors have full hidden dimension because they're all-reduced after the RowParallelLinear modules (down_proj, o_proj). Steering, patching, and ablation at the layer level use the exact same patterns as tp=1.

**Sub-module access crashes the engine with tp=2.** Accessing outputs from `qkv_proj`, `gate_up_proj`, `down_proj`, or `o_proj` inside a trace causes the vLLM engine to crash. This is a significant additional gap — with tp=1, these sub-modules are accessible (with the tuple/merged caveats documented above). With tp=2, the interleaver's hooks don't correctly handle the sharded execution across workers.

**Engine crashes are unrecoverable.** Once a sub-module access crashes the engine with tp=2, all subsequent traces fail with `EngineDeadError`. The vLLM process must be restarted.

### Expected sharding (architectural)

When sub-module access is fixed, these are the expected shapes:

| Module | tp=1 shape | tp=2 shape | Notes |
|--------|-----------|-----------|-------|
| `layer.output[0]` | `[tokens, 896]` | `[tokens, 896]` | All-reduced |
| `layer.output[1]` | `[tokens, 896]` | `[tokens, 896]` | Replicated |
| `qkv_proj.output` | `[tokens, 1152]` | `[tokens, 576]` | Sharded — half the heads per GPU |
| `gate_up_proj.output` | `[tokens, 9728]` | `[tokens, 4864]` | Sharded — half intermediate dim |
| `down_proj.output` | `(tensor, bias)` | `(tensor, bias)` | All-reduced after RowParallel |
| `o_proj.output` | `(tensor, bias)` | `(tensor, bias)` | All-reduced after RowParallel |

### tp=2 setup
```python
model = VLLM("Qwen/Qwen2.5-0.5B", tensor_parallel_size=2,
             gpu_memory_utilization=0.3, dtype=torch.float16, dispatch=True)
```

### Recommendations for tp=2

1. **Use layer-level interventions only** — these work reliably and have the same semantics as tp=1
2. **Avoid sub-module access** (qkv_proj, gate_up_proj, etc.) — currently crashes the engine
3. **Test with tp=1 first**, then verify the same intervention works with tp=2 at the layer level

---

## Quick Reference Table

| Operation | HF Pattern | vLLM Pattern | Gap |
|-----------|-----------|-------------|-----|
| **Save hidden state** | `layer.output[0].save()` | `(layer.output[0].clone() + layer.output[1].clone()).save()` | 1.1, 1.2 |
| **Get logits** | `model.lm_head.output` | `model.logits.output` | — |
| **Get layer input** | `layer.input` | `args, _ = layer.inputs; args[1] + args[2]` | 1.3 |
| **Last token** | `output[:, -1, :]` | `output[-1, :]` | 3.1 |
| **Zero layer** | `layer.output[0][:] = 0` | `layer.output = (zeros, zeros)` | 1.2 |
| **Steer** | `output[0][:, -1, :] += v` | `out0 = output[0].clone(); out0[-1,:] += v; layer.output = (out0, out1)` | 3.1, 1.2 |
| **Patch** | `output[0][:, -1, :] = target` | `layer.output = (target, zeros)` | 1.2, 3.1 |
| **Scale** | `output[0] *= s` | `layer.output = (out0*s, out1*s)` | 1.2 |
| **LayerNorm out** | `norm(x)` → tensor | `norm(x, residual)` → tuple | 1.4 |
| **gate/up proj** | `mlp.gate_proj`, `mlp.up_proj` | `mlp.gate_up_proj` (merged) | 2.1 |
| **q/k/v proj** | `attn.q_proj`, `attn.k_proj`, `attn.v_proj` | `attn.qkv_proj` (merged) | 2.2 |
| **down/o proj out** | tensor | `(tensor, bias)` tuple | 2.3 |
| **Attn weights** | `attn.source.attention_interface_0.output[0]` | Not available | 3.2 |
| **Gradients** | `requires_grad_()` + `backward()` | Not available | 4.1 |
| **Module skip** | `layer.skip(value)` | Broken (engine crash) | 4.3 |
| **Generation** | `model.generate(max_new_tokens=N)` | `model.trace(max_tokens=N)` | — |
| **Source tracing** | Works on all modules | Works on Python wrappers only | 4.2 |

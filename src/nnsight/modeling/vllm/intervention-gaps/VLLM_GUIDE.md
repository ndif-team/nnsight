# NNsight vLLM Intervention Guide

This guide mirrors the [main nnsight documentation](../../../../CLAUDE.md) (written for HuggingFace Transformers) and shows how each pattern translates to the vLLM backend.

All examples tested on **Qwen/Qwen2-0.5B** with vLLM 0.15.1, nnsight on branch `vllm-intervention-gaps`.

---

## Hard Blocks (Not Available on vLLM)

These features are **fundamentally impossible** on the vLLM backend due to C/CUDA-level architectural decisions. If your workflow requires any of these, use the HF backend instead.

| Feature | Why it's blocked | Impact |
|---------|-----------------|--------|
| **Gradients / backward pass** | vLLM wraps all execution in `torch.inference_mode()`, which globally disables gradient tracking. `requires_grad_(True)` raises `RuntimeError`. | Integrated gradients, saliency maps, GradCAM, gradient-based attribution, and training probes on activations are all blocked. |
| **Attention weights** | PagedAttention computes attention entirely in C/CUDA. No Python-level attention weight tensor exists — `self_attn.output` returns only the post-attention hidden state. | Attention visualization, head pruning, induction head detection, and attention knockout are impossible. |
| **Source tracing into fused modules** | Fused kernels (`input_layernorm`, `act_fn`, PagedAttention) have trivial Python wrappers that delegate to C/CUDA. `.source` reveals only the single delegate call, not the internal computation. | Fine-grained intervention inside fused modules is impossible. Non-fused modules (e.g. `self_attn`) work fine. |

---

## Remaining Differences (with compat layer)

Even with the compatibility layer (see below), these differences between HF and vLLM remain:

| Difference | HF | vLLM | Workaround |
|---|---|---|---|
| **Tensor layout** | 2D `[seq, hidden]` inside invokes | 2D `[tokens, hidden]` | Use `[-1, :]` not `[:, -1, :]` for last-token selection |
| **Logits access** | `model.lm_head.output` | `model.logits.output` | vLLM-specific module; `lm_head` may also work but `logits` is reliable |
| **Generation API** | `model.generate(max_new_tokens=N)` | `model.trace(max_tokens=N)` | Different method name and kwarg |
| **gate/up proj** | `mlp.gate_proj`, `mlp.up_proj` (separate) | `mlp.gate_up_proj` (merged) | `gate, up = output.chunk(2, dim=-1)` |
| **q/k/v proj** | `attn.q_proj`, `attn.k_proj`, `attn.v_proj` (separate) | `attn.qkv_proj` (merged) | `q, k, v = output.split([q_size, kv_size, kv_size], dim=-1)` |
| **LayerNorm inputs** | `RMSNorm(hidden)` — 1 arg | `fused_add_rms_norm(x, residual)` — 2 args | Access via `.inputs` and combine manually |
| **Numerical values** | PyTorch kernels | Fused CUDA kernels | Compare intervention *effects* (deltas), not absolute values |

---

## Compatibility Layer (auto-enabled)

NNsight includes a **transparent compatibility layer** (`VLLMBatcher` in `modeling/vllm/batching.py`) that automatically transforms vLLM-native values to match HuggingFace semantics. With the compat layer enabled (the default), most HF-style intervention code works on vLLM **without modification**.

### Why it exists

vLLM's internal architecture differs from HuggingFace Transformers in ways that would break standard intervention code:

- **Dual residual stream**: Decoder layers return `(mlp_output, residual)` as separate tensors instead of HF's `(hidden_states,)` combined 1-tuple.
- **In-place mutation**: `fused_add_rms_norm` mutates tensors after hooks fire, silently corrupting `.save()`'d references.
- **Different tuple formats**: `RMSNorm` returns `(normalized, residual)` tuples; `RowParallelLinear` returns `(output, bias)` tuples.
- **Int64 layer inputs**: Decoder layer `.input` returns position IDs, not float hidden states.

The compat layer masks all of these so users write the same code for both backends.

### How it works

The compat layer uses a **two-layer defense**:

1. **Detection** (once at model init): Heuristic checks flag modules as candidates for transformation — `'residual' in forward()` for decoder layers, `isinstance` for RMSNorm and RowParallelLinear. Unknown architectures fall through with no transform.

2. **Runtime guard** (every hook call): Before transforming, the code verifies the actual value matches the expected format (e.g. 2-tuple of tensors). If it doesn't match, the transform is silently skipped and a warning is logged. This prevents false positives from corrupting values.

```
PyTorch hook fires with raw vLLM output
  → pre_user_transform:  vLLM (mlp_out, residual) → HF (combined,)
    → user code sees HF-compatible values
  → post_user_transform: HF (combined,) → vLLM (combined.clone(), zeros)
PyTorch receives vLLM-compatible output
```

### What it auto-fixes

| Gap | Raw vLLM behavior | After compat layer | How |
|-----|-------------------|-------------------|-----|
| 1.1 | `.save()` silently corrupted by fused kernels | `.save()` is mutation-safe | Clones on both pre and post transform |
| 1.2 | `layer.output` is `(mlp_out, residual)` 2-tuple | `layer.output` is `(combined,)` 1-tuple | Combines streams, decomposes back |
| 1.3 | `layer.input` is int64 position IDs | `layer.input` is float hidden states | Filters positions, combines hidden + residual |
| 1.4 | `norm.output` is `(normalized, residual)` tuple | `norm.output` is a single tensor | Unwraps tuple, re-wraps on return |
| 2.3 | `down_proj.output` is `(tensor, bias)` tuple | `down_proj.output` is a single tensor | Unwraps tuple, re-wraps on return |
| 4.3 | `layer.skip(value)` crashes engine | `layer.skip(value)` works | Decomposes skip value to `(zeros, value)` for fused norm |

### Model detection

The compat layer auto-detects which modules need transforms:
- **Decoder layers**: detected via `'residual' in forward()` parameters — works for Qwen2, Llama, Mistral, Mixtral, Gemma2, DeepSeek-V2, Phi3
- **GPT2Block**: has no `residual` param → no transform applied (correct — GPT2 uses single-stream)
- **RMSNorm**: detected via `isinstance(module, vllm...RMSNorm)`
- **RowParallelLinear**: detected via `isinstance(module, RowParallelLinear)`

Unknown architectures fall through with **no transform** (same as raw vLLM behavior). If detection flags a module but the runtime value doesn't match expectations, the transform is skipped and a warning is logged.

### inference_mode handling

vLLM wraps all execution in `torch.inference_mode()`. The compat layer briefly exits inference mode (`with torch.inference_mode(False):`) for exactly 2 tensor operations (clone + add) when creating user-facing combined tensors. This is necessary so users can do in-place modifications like `output[0][:] = 0`. The scope is minimal (thread-local bool toggle), does not affect vLLM's forward pass (already completed), and vLLM never checks `is_inference_mode_enabled()` at runtime.

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
model = LanguageModel("Qwen/Qwen2-0.5B", device_map="auto", dispatch=True)
```

### vLLM
```python
from nnsight.modeling.vllm import VLLM
model = VLLM("Qwen/Qwen2-0.5B", tensor_parallel_size=1,
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

### vLLM (with compat layer — same code works)
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    hidden = model.model.layers[-1].output[0].save()
# hidden is the combined hidden state, shape [total_tokens, hidden_dim]
```

The compat layer automatically combines the dual residual streams and protects against in-place mutation. `output[0]` returns the full combined hidden state (equivalent to HF), and `.save()` is mutation-safe.

**Only remaining difference:** shape is 2D `[tokens, hidden]` instead of 3D `[batch, seq, hidden]` (Gap 3.1).

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

### vLLM (with compat layer — same code works)
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    layer_input = model.model.layers[4].input.save()
    # Returns float hidden states, shape [total_tokens, hidden_dim]
```

The compat layer transforms the input from `(positions, hidden_states, residual)` to `(combined_hidden_states,)`, so `.input` returns the combined float hidden state just like HF. For layer 0 where `residual=None`, `.input` returns just `hidden_states`.

---

## Saving Values

### HF pattern
```python
with model.trace("Hello"):
    hidden = model.transformer.h[0].output[0].save()
```

### vLLM (with compat layer — same code works)
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    hidden = model.model.layers[0].output[0].save()
```

The compat layer auto-clones decoder layer outputs, so `.save()` is mutation-safe. No manual `.clone()` needed.

**Background (Gap 1.1):** Without the compat layer, vLLM's fused CUDA kernels (`fused_add_rms_norm`) mutate tensors in-place after hooks fire, silently corrupting saved values (max_diff up to 1013.81).

---

## Modifying Activations — Ablation

### HF pattern
```python
with model.trace("Hello"):
    model.transformer.h[0].output[0][:] = 0    # Zero the hidden state
```

### vLLM (with compat layer — same code works)
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    model.model.layers[0].output[0][:] = 0    # Zero the combined hidden state
```

The compat layer presents a `(combined,)` 1-tuple. In-place zeroing modifies the combined tensor, and `post_user_transform` decomposes it back to `(zeros, zeros)` for both streams — equivalent to zeroing the full hidden state.

---

## Modifying Activations — Steering

### HF pattern
```python
steering_vector = torch.randn(768)  # Pre-computed direction

with model.trace("Hello"):
    model.transformer.h[10].output[0][:, -1, :] += steering_vector * 0.5
```

### vLLM (with compat layer — nearly identical)
```python
steering_vector = torch.randn(896, dtype=torch.bfloat16).cuda() * 0.5

with model.trace("Hello", temperature=0.0, top_p=1):
    model.model.layers[10].output[0][-1, :] += steering_vector  # 2D: [-1, :] not [:, -1, :]
```

The compat layer handles the dual-stream decomposition automatically. The only remaining difference is **2D indexing** (`[-1, :]` instead of `[:, -1, :]`) due to Gap 3.1.

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

### vLLM (with compat layer — nearly identical)
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in", temperature=0.0, top_p=1):
        clean_hs = model.model.layers[5].output[0][-1, :].save()   # 2D indexing
        barrier()
    with tracer.invoke("The Colosseum is in", temperature=0.0, top_p=1):
        barrier()
        model.model.layers[5].output[0][-1, :] = clean_hs          # 2D indexing
        logits = model.logits.output.save()
```

The compat layer handles combining/decomposing the dual streams, cloning, and tuple management. Only remaining difference: `[-1, :]` (2D) instead of `[:, -1, :]` (3D), and `model.logits.output` instead of `model.lm_head.output`.

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

### vLLM (with compat layer — nearly identical)
```python
with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
    for i in range(24):
        hs = model.model.layers[i].output[0]          # Already combined by compat
        normed = model.model.norm(hs)                  # Already unwrapped by compat
        if isinstance(normed, tuple):                   # Defensive check for norm called as function
            normed = normed[0]
        logits = model.lm_head(normed)
        tokens = logits.argmax(dim=-1).save()
        print(f"Layer {i}:", model.tokenizer.decode(tokens[-1]))  # 2D: [-1] not [0][-1]
```

The compat layer handles Gap 1.2 (output combining) and Gap 1.4 (norm unwrapping) automatically. The `isinstance` check is a defensive guard for when `model.model.norm(hs)` is called as a function rather than accessed via `.output` (function calls bypass compat hooks). Only remaining difference: `tokens[-1]` (2D) instead of `tokens[0][-1]` (3D).

**Verified:** Logit lens produces "Paris" at deeper layers for "The Eiffel Tower is in".

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

### vLLM (with compat layer — same code works)
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    normed = model.model.layers[0].input_layernorm.output  # Returns single tensor
```

The compat layer auto-unwraps the fused RMSNorm's `(normalized, residual)` tuple and returns just the normalized tensor, matching HF behavior.

**Gap 1.5 (inputs):** The input to vLLM's layernorm still differs (2 args vs 1). This is not transformed by the compat layer.

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

**vLLM (with compat layer — same code works):** The compat layer auto-unwraps the `(output, bias)` tuple, so `down_proj.output` returns a single tensor just like HF.

```python
# Works directly — compat layer strips the bias tuple
actual_output = model.model.layers[0].mlp.down_proj.output
```

**Background (Gap 2.3):** Without the compat layer, vLLM's `RowParallelLinear` returns `(output, output_bias)` 2-tuples.

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

### vLLM (with compat layer — same code works)
```python
with model.trace("Hello", temperature=0.0, top_p=1):
    layer0_out = model.model.layers[0].output
    model.model.layers[1].skip(layer0_out)
```

The compat layer's `post_user_transform` detects skip values and decomposes them into the `(zeros, value)` format that vLLM's `fused_add_rms_norm` expects, so `fused_add_rms_norm(0, v) = rms_norm(v)`.

**Background (Gap 4.3):** Without the compat layer, `skip()` crashes the vLLM engine with `EngineDeadError` because fused norm expects `(hidden_states, residual)` pairs.

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

### vLLM (with compat layer — same code works)
```python
with model.trace("Hello", temperature=0.0, top_p=1) as tracer:
    cache = tracer.cache(modules=[model.model.layers[0]])

out = cache.model.layers[0].output
# With compat layer: out is a (combined_hidden_states,) 1-tuple, same as HF
hidden = out[0]  # Already combined, already clone-safe
```

The compat layer transforms cached values the same way as `.output` — dual streams are combined and mutation-protected.

**Note:** The cache captures values after the compat `pre_user_transform`, so cached decoder layer outputs are already in HF format.

---

## Tensor Parallelism (tp=2)

With `tensor_parallel_size=2`, vLLM shards certain modules across GPUs. This introduces additional differences from tp=1.

### Verified: tp=2 on Qwen/Qwen2-0.5B (GPUs 0,7)

> **Warning:** Accessing sub-module outputs (`qkv_proj`, `gate_up_proj`, `down_proj`, `o_proj`) with tp=2 **crashes the vLLM engine unrecoverably**. All subsequent traces fail with `EngineDeadError`. Stick to layer-level interventions only.

| Test | tp=1 | tp=2 | Status |
|------|------|------|--------|
| `layer.output[0]` shape | `[2, 896]` | `[2, 896]` | **Same** — combined hidden state (compat layer) |
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
| `layer.output[0]` | `[tokens, 896]` | `[tokens, 896]` | Combined hidden state (compat layer) |
| `qkv_proj.output` | `[tokens, 1152]` | `[tokens, 576]` | Sharded — half the heads per GPU |
| `gate_up_proj.output` | `[tokens, 9728]` | `[tokens, 4864]` | Sharded — half intermediate dim |
| `down_proj.output` | tensor | tensor | Compat layer unwraps `(output, bias)` tuple |
| `o_proj.output` | tensor | tensor | Compat layer unwraps `(output, bias)` tuple |

### tp=2 setup
```python
model = VLLM("Qwen/Qwen2-0.5B", tensor_parallel_size=2,
             gpu_memory_utilization=0.3, dtype=torch.float16, dispatch=True)
```

### Recommendations for tp=2

1. **Use layer-level interventions only** — these work reliably and have the same semantics as tp=1
2. **Avoid sub-module access** (qkv_proj, gate_up_proj, etc.) — currently crashes the engine
3. **Test with tp=1 first**, then verify the same intervention works with tp=2 at the layer level

---

## Quick Reference Table

### With compat layer (default)

| Operation | HF Pattern | vLLM Pattern (compat) | Notes |
|-----------|-----------|----------------------|-------|
| **Save hidden state** | `layer.output[0].save()` | **Same** | Auto-combined, auto-cloned |
| **Get layer input** | `layer.input` | **Same** | Auto-combined hidden states |
| **Zero layer** | `layer.output[0][:] = 0` | **Same** | Auto-decomposed to both streams |
| **Steer** | `output[0][:, -1, :] += v` | `output[0][-1, :] += v` | Only diff: 2D indexing (Gap 3.1) |
| **Patch** | `output[0][:, -1, :] = target` | `output[0][-1, :] = target` | Only diff: 2D indexing |
| **Scale** | `output[0] *= s` | **Same** | Auto-decomposed |
| **LayerNorm out** | `norm(x)` → tensor | **Same** | Auto-unwrapped |
| **down/o proj out** | tensor | **Same** | Auto-unwrapped |
| **Module skip** | `layer.skip(value)` | **Same** | Auto-decomposed for fused norm |
| **Get logits** | `model.lm_head.output` | `model.logits.output` | Different module path |
| **Generation** | `model.generate(max_new_tokens=N)` | `model.trace(max_tokens=N)` | Different API |
| **gate/up proj** | `mlp.gate_proj`, `mlp.up_proj` | `mlp.gate_up_proj` (merged) | Use `.chunk(2)` (Gap 2.1) |
| **q/k/v proj** | `attn.q_proj`, `attn.k_proj`, `attn.v_proj` | `attn.qkv_proj` (merged) | Use `.split()` (Gap 2.2) |
| **Attn weights** | `attn.source...output[0]` | Not available | Hard block (Gap 3.2) |
| **Gradients** | `requires_grad_()` + `backward()` | Not available | Hard block (Gap 4.1) |
| **Source tracing** | Works on all modules | Python wrappers only | Hard block (Gap 4.2) |

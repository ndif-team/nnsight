# vLLM ↔ HuggingFace Intervention Gap Report

**Date:** 2026-03-24
**Model:** Qwen/Qwen2-0.5B
**vLLM version:** 0.15.1
**nnsight branch:** worktree-vllm_debug
**Hardware:** 8× NVIDIA A100 (vLLM on GPU 0, HF on GPU 1)

## Summary

**13 gaps documented. 6 auto-fixed by the HF-compatibility layer** (`VLLMBatcher.pre_user_transform` / `post_user_transform`), 7 remain as documented differences or hard blocks.

| Gap | Description | Compat Status |
|-----|-------------|---------------|
| 1.1 | In-place mutation corrupts `.save()` | **AUTO-FIXED** — pre/post transforms clone |
| 1.2 | Decoder layer output: `(mlp, res)` vs `(combined,)` | **AUTO-FIXED** — streams combined |
| 1.3 | Decoder layer `.input` returns int64 positions | **AUTO-FIXED** — transformed to float hidden states |
| 1.4 | LayerNorm output: tuple vs tensor | **AUTO-FIXED** — tuple unwrapped |
| 1.5 | LayerNorm input semantics differ | DOCUMENTED |
| 2.1 | MLP submodule layout (merged gate\_up\_proj) | DOCUMENTED |
| 2.2 | Attention submodule layout (merged qkv\_proj) | DOCUMENTED |
| 2.3 | RowParallelLinear returns tuple | **AUTO-FIXED** — tuple unwrapped |
| 3.1 | Flat batch dimension [total\_tokens, hidden] | DOCUMENTED |
| 3.2 | PagedAttention: no attention weights | HARD BLOCK |
| 4.1 | Gradients blocked by inference\_mode | HARD BLOCK |
| 4.2 | Source tracing into fused kernels | HARD BLOCK |
| 4.3 | Module skip breaks fused norm | **AUTO-FIXED** — skip value decomposed |

The gap descriptions below document the **underlying architectural differences** that still exist in vLLM. The compat layer masks these transparently so HF-style code works without modification.

---

## Group 1: Activation Semantics

### Gap 1.1 — In-place mutation corrupts `.save()` — **AUTO-FIXED**

**Root cause:** vLLM's `fused_add_rms_norm` mutates tensors in-place after nnsight hooks fire. `.save()` stores a reference, so the saved value silently becomes the post-mutation value.

**Evidence (raw, without compat layer):**
- `layer.output[0]` ref vs clone: diff = **64.62**
- `layer.output[1]` ref vs clone: diff = **1013.81**
- `mlp.output` and `self_attn.output`: unaffected (not mutated downstream)

**Compat fix:** `pre_user_transform` creates the combined tensor via `clone() + clone()`, giving the user a fresh tensor. `post_user_transform` clones again before returning to vLLM, so fused norm mutations don't propagate back.

**Impact without compat:** Silent data corruption in probing, activation patching, logit lens.

---

### Gap 1.2 — Decoder layer output semantics differ — **AUTO-FIXED**

**Root cause:** vLLM maintains a dual residual stream. Decoder layers return `(hidden_states, residual)` as separate tuple elements. HF returns `(hidden_states + residual,)` combined.

**Evidence (raw, without compat layer):**
- vLLM: output is 2-tuple; `output[0] == mlp.output` (cosine = 1.0)
- HF: output is 1-tuple; `output[0] != mlp.output` (cosine = 0.45) because it includes the residual

**Compat fix:** `pre_user_transform` combines `(mlp_out, residual)` into `(mlp_out + residual,)` 1-tuple. `post_user_transform` decomposes back to `(combined.clone(), zeros)` for vLLM.

**Impact without compat:** Logit lens, steering vectors, activation patching all operate on the wrong representation.

---

### Gap 1.3 — Decoder layer input semantics differ — **AUTO-FIXED**

**Root cause:** vLLM decoder signature is `forward(positions, hidden_states, residual, ...)`. The first positional arg (`.input`) is int64 position IDs, not float hidden states.

**Evidence (raw, without compat layer):**
- vLLM `.input` dtype: `torch.int64`, values: `[0, 1, 2, 3, 4]`
- HF `.input` dtype: `torch.float32`, shape: `[1, 5, 896]`

**Compat fix:** `pre_user_transform` filters out `positions`, combines `hidden_states + residual` into a single float tensor, and presents it as `(combined_hidden_states,)`. For layer 0 where `residual=None`, presents just `hidden_states`.

**Impact without compat:** Arithmetic on `.input` silently promotes int to float; cross-layer analysis operates on wrong data.

---

### Gap 1.4 — LayerNorm output: tensor vs tuple — **AUTO-FIXED**

**Root cause:** vLLM's fused `fused_add_rms_norm` returns `(normalized, new_residual)` tuple. HF returns a single tensor.

**Evidence (raw, without compat layer):**
- vLLM `input_layernorm.output`: is tuple = True
- HF `input_layernorm.output`: is tensor = True

**Compat fix:** `pre_user_transform` detects `RMSNorm` modules and unwraps `(normalized, residual)` to just `normalized`. `post_user_transform` wraps back to tuple if the original was a tuple.

**Impact without compat:** Logit lens fails (can't pass tuple to lm_head). Layernorm output analysis breaks.

---

### Gap 1.5 — LayerNorm input semantics differ

**Root cause:** vLLM uses `fused_add_rms_norm(x, residual)` (2 args). HF uses `RMSNorm(hidden)` (1 arg).

**Evidence:**
- vLLM `input_layernorm`: num_args = 2
- HF `input_layernorm`: num_args = 1

**Impact:** Reading/modifying layernorm inputs gets incomplete value (raw output without residual component).

---

## Group 2: Module Architecture

### Gap 2.1 — MLP submodule layout (merged gate\_up\_proj)

**Root cause:** vLLM uses a single `MergedColumnParallelLinear` (`gate_up_proj`) instead of separate `gate_proj` and `up_proj`.

**Evidence:**
- vLLM: `gate_proj=False`, `up_proj=False`, `gate_up_proj=True`
- HF: `gate_proj=True`, `up_proj=True`, `gate_up_proj=False`

**Impact:** Cannot analyze/ablate gate vs up projections separately. SAE training on individual projections impossible.

---

### Gap 2.2 — Attention submodule layout (merged qkv\_proj)

**Root cause:** vLLM uses a single `QKVParallelLinear` (`qkv_proj`) instead of separate `q_proj`, `k_proj`, `v_proj`.

**Evidence:**
- vLLM: `q_proj=False`, `k_proj=False`, `v_proj=False`, `qkv_proj=True`
- HF: `q_proj=True`, `k_proj=True`, `v_proj=True`, `qkv_proj=False`

**Impact:** Attention head analysis, key/value editing, Q/K/V-specific interventions all require manual tensor splitting.

---

### Gap 2.3 — RowParallelLinear returns tuple — **AUTO-FIXED**

**Root cause:** vLLM's `RowParallelLinear` returns `(output, output_bias)` 2-tuple. HF `Linear` returns a single tensor.

**Evidence (raw, without compat layer):**
- vLLM: `down_proj` tuple = True, `o_proj` tuple = True
- HF: `down_proj` tuple = False, `o_proj` tuple = False

**Compat fix:** `pre_user_transform` detects `RowParallelLinear` modules and unwraps `(output, bias)` to just `output`. `post_user_transform` wraps back to tuple.

**Impact without compat:** `down_proj.output * 2` fails. Any intervention needs tuple unpacking.

---

## Group 3: Data Layout & Accessibility

### Gap 3.1 — Flat batch dimension [total\_tokens, hidden]

**Root cause:** vLLM uses continuous batching with a flat 2D layout `[total_tokens, hidden]`. HF natively uses padded 3D `[batch, seq, hidden]`.

**Evidence:**
- vLLM: ndim = 2, shape = `[7, 896]` — tokens from all sequences packed flat
- HF: ndim = 2, shape = `[7, 896]` — nnsight's batcher narrows the batch dimension even for single-input traces (implicit invoke)

**Clarification:** Inside nnsight traces, both backends produce 2D tensors because nnsight creates an implicit invoke that narrows the batch dimension. The gap manifests in two ways: (1) **Doc examples use 3D indexing** like `[:, -1, :]` which assumes a `[batch, seq, hidden]` layout — these fail on vLLM because the native format is flat `[total_tokens, hidden]` with no sequence dimension; (2) **Multi-sequence batching** in vLLM packs all tokens contiguously with no padding, so token boundary tracking requires out-of-band knowledge of sequence lengths. Use `[-1, :]` for last-token selection on both backends inside traces.

**Impact:** Positional indexing like `[:, -1, :]` (last token) or `[:, pos, :]` breaks. Token targeting requires knowing token boundaries in the flat layout.

---

### Gap 3.2 — PagedAttention: no attention weights

**Root cause:** vLLM's PagedAttention computes attention entirely in C/CUDA. No Python-level attention weight tensor exists.

**Evidence:**
- vLLM attn output: single tensor, shape `[7, 896]`
- HF attn output: tuple, length 2 (hidden\_states + attention\_weights)

**Impact:** Attention visualization, head pruning, induction head detection, attention knockout all impossible. Fundamental architectural limitation.

---

## Group 4: Advanced Features

### Gap 4.1 — Gradients blocked by inference\_mode

**Root cause:** vLLM wraps all execution in `torch.inference_mode()`, disabling gradient tracking globally.

**Evidence:**
- vLLM: `requires_grad_()` FAILED (RuntimeError from inference\_mode), `backward()` FAILED
- HF: `requires_grad_()` OK, `backward()` FAILED (unrelated nnsight interleaver issue — grad not captured for this model/layer combination; the key asymmetry is that `requires_grad_()` itself succeeds on HF, proving no inference\_mode barrier)

**Impact:** Integrated gradients, GradCAM, saliency maps, gradient-based attribution, probe training all impossible.

---

### Gap 4.2 — Source tracing into fused kernels

**Root cause:** vLLM's fused modules (`input_layernorm`, `act_fn`, PagedAttention) have trivial Python wrappers that delegate to C/CUDA. `.source` reveals only the delegate call.

**Evidence:**
- vLLM `input_layernorm`: 1 op (175 chars) — single delegate
- HF `input_layernorm`: 6 ops — full Python-visible computation
- vLLM `act_fn`: 1 op vs HF: 2 ops

**Impact:** Fine-grained intervention inside fused modules impossible. Intervention granularity capped at module level.

---

### Gap 4.3 — Module skip breaks fused norm — **AUTO-FIXED**

**Root cause:** vLLM decoder layers return `(hidden_states, residual)` tuple. Skipping with a single tensor fails because the next layer's `fused_add_rms_norm` expects the `(x, residual)` pair.

**Evidence (raw, without compat layer):**
- vLLM `skip(single_tensor)`: FAILED (EngineDeadError — unrecoverable engine crash)
- vLLM `skip(tuple)`: FAILED (EngineDeadError — engine already dead from first attempt)
- HF `skip(single_tensor)`: FAILED (shape mismatch — HF layers also expect tuple input)
- HF `skip(tuple)`: OK

**Compat fix:** `post_user_transform` detects skip values on decoder layers and decomposes the skip value into `(zeros, value)`, so `fused_add_rms_norm(0, v) = rms_norm(v)`.

**Note:** `skip(single_tensor)` fails on both backends without the compat layer, but for different reasons: HF raises a shape error (recoverable), while vLLM crashes the engine (unrecoverable — all subsequent traces fail with `EngineDeadError`, requiring process restart).

**Impact without compat:** Layer ablation studies blocked on vLLM.

---

## Architectural Root Causes

All 13 gaps stem from four architectural decisions in vLLM:

1. **Fused CUDA kernels** — `fused_add_rms_norm`, `SiluAndMul`, PagedAttention mutate tensors in-place, return tuples, and hide internals from Python hooks. (Gaps 1.1✓, 1.4✓, 1.5, 3.2, 4.2, 4.3✓)

2. **Dual residual stream** — Hidden states and residual are kept as separate tensors throughout the forward pass, unlike HF which combines them. (Gaps 1.2✓, 1.3✓)

3. **Merged linear layers** — `ColumnParallelLinear` (gate\_up\_proj, qkv\_proj) and `RowParallelLinear` (returning tuples) for tensor parallelism. (Gaps 2.1, 2.2, 2.3✓)

4. **Continuous batching + inference\_mode** — Flat `[total_tokens, hidden]` layout and `torch.inference_mode()` wrapper for performance. (Gaps 3.1, 4.1)

✓ = auto-fixed by HF-compatibility layer

---

## Fixability Assessment

| Category | Status | Implementation |
|----------|--------|----------------|
| **Gap 1.1** (mutation) | **FIXED** | `pre_user_transform` clones; `post_user_transform` clones before returning to vLLM |
| **Gap 1.2** (dual residual output) | **FIXED** | `pre_user_transform` combines `(mlp, res)` → `(combined,)` 1-tuple |
| **Gap 1.3** (input positions) | **FIXED** | `pre_user_transform` filters positions, combines hidden + residual |
| **Gap 1.4** (norm tuple) | **FIXED** | `pre_user_transform` unwraps `(normalized, residual)` → `normalized` |
| **Gap 1.5** (norm input) | Documented | Sub-module input semantics not transformed |
| **Gaps 2.1–2.2** (merged modules) | Not fixable | Fundamental vLLM architecture; use `.chunk()` / `.split()` |
| **Gap 2.3** (RowParallel tuple) | **FIXED** | `pre_user_transform` unwraps `(output, bias)` → `output` |
| **Gap 3.1** (flat layout) | Documented | Use `[-1, :]` not `[:, -1, :]`; nnsight invoker already narrows per-invoke |
| **Gap 3.2** (no attn weights) | Not fixable | PagedAttention is a fused CUDA kernel; no Python-level access |
| **Gap 4.1** (no gradients) | Not fixable | `inference_mode` is set by vLLM engine; disabling it would break vLLM internals |
| **Gap 4.2** (fused source) | Not fixable | Computation happens in C/CUDA; Python-level `.source` can't reach it |
| **Gap 4.3** (skip breaks) | **FIXED** | `post_user_transform` decomposes skip value to `(zeros, value)` for fused norm |

---

## Intervention Patterns (with compat layer)

The HF-compatibility layer (`VLLMBatcher`) auto-fixes gaps 1.1–1.4, 2.3, and 4.3, so **standard HF-style intervention code works on vLLM without modification** (except for 2D indexing from Gap 3.1).

### How the compat layer works

```
PyTorch hook fires with raw vLLM output
  → pre_user_transform:  vLLM (mlp_out, residual) → HF (combined,)
    → user code sees HF-compatible values
  → post_user_transform: HF (combined,) → vLLM (combined.clone(), zeros)
PyTorch receives vLLM-compatible output
```

### HF-compatible patterns (use these)

| Operation | Pattern | Notes |
|-----------|---------|-------|
| **Save** | `layer.output[0].save()` | Auto-cloned, mutation-safe |
| **Steer** | `layer.output[0][-1, :] += v` | 2D indexing (Gap 3.1) |
| **Patch** | `layer.output[0][-1, :] = target` | 2D indexing |
| **Ablate** | `layer.output[0][:] = 0` | Auto-decomposes to both streams |
| **Scale** | `layer.output[0][:] *= s` | Auto-decomposes to both streams |
| **Skip** | `layer.skip(value)` | Auto-decomposes for fused norm |
| **LayerNorm** | `norm.output` → tensor | Auto-unwrapped from tuple |
| **down/o proj** | `proj.output` → tensor | Auto-unwrapped from tuple |
| **Layer input** | `layer.input` → float tensor | Auto-combined hidden states |

### Remaining differences (not auto-fixed)

1. **2D layout** `[tokens, hidden]` not 3D `[batch, seq, hidden]` — index with `[-1, :]` not `[:, -1, :]` (Gap 3.1)
2. **Logits** — use `model.logits.output` instead of `model.lm_head.output`
3. **Generation** — use `model.trace(max_tokens=N)` instead of `model.generate(max_new_tokens=N)`
4. **Merged projections** — `gate_up_proj` (Gap 2.1), `qkv_proj` (Gap 2.2) require `.chunk()` / `.split()`
5. **Baseline logit gap** — vLLM and HF use different kernels (fused vs separate), producing slightly different numerical results. Compare intervention *effects* (delta from baseline), not absolute values.

### Background: vLLM's dual residual stream (hidden by compat layer)

Under the hood, vLLM decoder layers still return `(mlp_output, residual)` as separate tensors. The compat layer combines them into `(mlp_output + residual,)` before user code sees them, and decomposes back to `(combined.clone(), zeros)` afterward. This is mathematically equivalent because the next layer's `fused_add_rms_norm` computes `RMSNorm(stream0 + stream1)` — only the sum matters.

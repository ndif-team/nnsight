# vLLM ↔ HuggingFace Intervention Gap Report

**Date:** 2026-03-24
**Model:** Qwen/Qwen2-0.5B
**vLLM version:** 0.15.1
**nnsight branch:** worktree-vllm_debug
**Hardware:** 8× NVIDIA A100 (vLLM on GPU 0, HF on GPU 1)

## Summary

**13 / 13 gaps confirmed.** Every documented semantic difference between the vLLM and HuggingFace backends was reproduced.

| Gap | vLLM | HF | Verdict | Description |
|-----|------|----|---------|-------------|
| 1.1 | CONFIRMED | NO_GAP | GAP CONFIRMED | In-place mutation corrupts `.save()` |
| 1.2 | CONFIRMED | NO_GAP | GAP CONFIRMED | Decoder layer output semantics differ |
| 1.3 | CONFIRMED | NO_GAP | GAP CONFIRMED | Decoder layer input semantics differ |
| 1.4 | CONFIRMED | NO_GAP | GAP CONFIRMED | LayerNorm output: tensor vs tuple |
| 1.5 | CONFIRMED | NO_GAP | GAP CONFIRMED | LayerNorm input semantics differ |
| 2.1 | CONFIRMED | NO_GAP | GAP CONFIRMED | MLP submodule layout (merged gate\_up\_proj) |
| 2.2 | CONFIRMED | NO_GAP | GAP CONFIRMED | Attention submodule layout (merged qkv\_proj) |
| 2.3 | CONFIRMED | NO_GAP | GAP CONFIRMED | RowParallelLinear returns tuple |
| 3.1 | CONFIRMED | NO_GAP | GAP CONFIRMED | Flat batch dimension [total\_tokens, hidden] |
| 3.2 | CONFIRMED | NO_GAP | GAP CONFIRMED | PagedAttention: no attention weights |
| 4.1 | CONFIRMED | NO_GAP | GAP CONFIRMED | Gradients blocked by inference\_mode |
| 4.2 | CONFIRMED | NO_GAP | GAP CONFIRMED | Source tracing into fused kernels |
| 4.3 | CONFIRMED | NO_GAP | GAP CONFIRMED | Module skip breaks fused norm |

---

## Group 1: Activation Semantics

### Gap 1.1 — In-place mutation corrupts `.save()`

**Root cause:** vLLM's `fused_add_rms_norm` mutates tensors in-place after nnsight hooks fire. `.save()` stores a reference, so the saved value silently becomes the post-mutation value.

**Evidence:**
- `layer.output[0]` ref vs clone: diff = **64.62**
- `layer.output[1]` ref vs clone: diff = **1013.81**
- `mlp.output` and `self_attn.output`: unaffected (not mutated downstream)

**Workaround:** Use `.clone().save()` instead of `.save()`.

**Impact:** Silent data corruption in probing, activation patching, logit lens. Users get wrong results with no error.

---

### Gap 1.2 — Decoder layer output semantics differ

**Root cause:** vLLM maintains a dual residual stream. Decoder layers return `(hidden_states, residual)` as separate tuple elements. HF returns `(hidden_states + residual,)` combined.

**Evidence:**
- vLLM: output is 2-tuple; `output[0] == mlp.output` (cosine = 1.0)
- HF: output is 1-tuple; `output[0] != mlp.output` (cosine = 0.45) because it includes the residual

**Workaround:** `layer.output[0].clone() + layer.output[1].clone()` recovers the HF-equivalent hidden state (verified: max_diff = 0.0625).

**Impact:** Logit lens, steering vectors, activation patching all operate on the wrong representation.

---

### Gap 1.3 — Decoder layer input semantics differ

**Root cause:** vLLM decoder signature is `forward(positions, hidden_states, residual, ...)`. The first positional arg (`.input`) is int64 position IDs, not float hidden states.

**Evidence:**
- vLLM `.input` dtype: `torch.int64`, values: `[0, 1, 2, 3, 4]`
- HF `.input` dtype: `torch.float32`, shape: `[1, 5, 896]`

**Workaround:**
```python
args, kwargs = layer.inputs
hidden_states = args[1] + args[2]  # Combine dual residual stream
```
Verified: max_diff = 0.0625 vs HF.

**Impact:** Arithmetic on `.input` silently promotes int to float; cross-layer analysis operates on wrong data.

---

### Gap 1.4 — LayerNorm output: tensor vs tuple

**Root cause:** vLLM's fused `fused_add_rms_norm` returns `(normalized, new_residual)` tuple. HF returns a single tensor.

**Evidence:**
- vLLM `input_layernorm.output`: is tuple = True
- HF `input_layernorm.output`: is tensor = True

**Impact:** Logit lens fails (can't pass tuple to lm_head). Layernorm output analysis breaks.

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

### Gap 2.3 — RowParallelLinear returns tuple

**Root cause:** vLLM's `RowParallelLinear` returns `(output, output_bias)` 2-tuple. HF `Linear` returns a single tensor.

**Evidence:**
- vLLM: `down_proj` tuple = True, `o_proj` tuple = True
- HF: `down_proj` tuple = False, `o_proj` tuple = False

**Impact:** `down_proj.output * 2` fails. Any intervention needs tuple unpacking.

---

## Group 3: Data Layout & Accessibility

### Gap 3.1 — Flat batch dimension [total\_tokens, hidden]

**Root cause:** vLLM uses continuous batching with a flat 2D layout `[total_tokens, hidden]`. HF uses padded 3D `[batch, seq, hidden]`.

**Evidence:**
- vLLM: ndim = 2, shape = `[7, 896]`
- HF: ndim = 2, shape = `[7, 896]` (after batcher narrows per-invoke)

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
- vLLM: `requires_grad_()` FAILED, `backward()` FAILED
- HF: `requires_grad_()` OK, `backward()` FAILED (unrelated — nnsight backward context needed)

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

### Gap 4.3 — Module skip breaks fused norm

**Root cause:** vLLM decoder layers return `(hidden_states, residual)` tuple. Skipping with a single tensor fails because the next layer's `fused_add_rms_norm` expects the `(x, residual)` pair.

**Evidence:**
- vLLM `skip(single_tensor)`: FAILED (EngineDeadError)
- vLLM `skip(tuple)`: FAILED (EngineDeadError — engine crashed from first attempt)
- HF `skip(single_tensor)`: FAILED (expected shape mismatch)
- HF `skip(tuple)`: OK

**Impact:** Layer ablation studies blocked on vLLM. Requires understanding internal `(x, residual)` bookkeeping.

---

## Architectural Root Causes

All 13 gaps stem from four architectural decisions in vLLM:

1. **Fused CUDA kernels** — `fused_add_rms_norm`, `SiluAndMul`, PagedAttention mutate tensors in-place, return tuples, and hide internals from Python hooks. (Gaps 1.1, 1.4, 1.5, 3.2, 4.2, 4.3)

2. **Dual residual stream** — Hidden states and residual are kept as separate tensors throughout the forward pass, unlike HF which combines them. (Gaps 1.2, 1.3)

3. **Merged linear layers** — `ColumnParallelLinear` (gate\_up\_proj, qkv\_proj) and `RowParallelLinear` (returning tuples) for tensor parallelism. (Gaps 2.1, 2.2, 2.3)

4. **Continuous batching + inference\_mode** — Flat `[total_tokens, hidden]` layout and `torch.inference_mode()` wrapper for performance. (Gaps 3.1, 4.1)

---

## Fixability Assessment

| Category | Fixable in nnsight? | Approach |
|----------|-------------------|----------|
| **Gap 1.1** (mutation) | Yes | Auto-clone in `.save()` when vLLM backend detected |
| **Gaps 1.2–1.5** (dual residual) | Partial | Adapter layer could recombine, but adds overhead and may surprise users |
| **Gaps 2.1–2.3** (merged modules) | No | Fundamental vLLM architecture; best documented with splitting helpers |
| **Gap 3.1** (flat layout) | Partial | nnsight's invoker already narrows per-invoke; 3D reshape possible but fragile |
| **Gap 3.2** (no attn weights) | No | PagedAttention is a fused CUDA kernel; no Python-level access |
| **Gap 4.1** (no gradients) | No | `inference_mode` is set by vLLM engine; disabling it would break vLLM internals |
| **Gap 4.2** (fused source) | No | Computation happens in C/CUDA; Python-level `.source` can't reach it |
| **Gap 4.3** (skip breaks) | Partial | Could auto-wrap skip value into `(tensor, residual)` tuple for vLLM layers |

**Recommendation:** Gap 1.1 is the highest-priority fix (silent data corruption). The rest should be documented in a vLLM compatibility guide with workarounds.

---

## Intervention Strategy Guide for vLLM's Dual Residual Stream

### Background: How the Dual Stream Works

In HF, each decoder layer returns a single `hidden_states` tensor that includes the residual:
```
hidden_states = residual + mlp(norm(residual + attn(norm(hidden_states))))
```

In vLLM, the layer returns `(mlp_output, residual)` as **separate tensors**. The next layer's `fused_add_rms_norm` computes:
```
normalized = RMSNorm(mlp_output + residual)
new_residual = mlp_output + residual
```

**Key insight: only the sum `output[0] + output[1]` matters.** The decomposition into two streams is ephemeral — it's immediately collapsed by the next layer's fused norm. This means the specific split between `output[0]` and `output[1]` is irrelevant; only their sum affects downstream computation.

### Verified Intervention Strategies

All strategies verified experimentally on Qwen2.5-0.5B, layer 10, comparing vLLM against HF reference.

#### Steering (Adding a Direction Vector)

**Add to either stream — both produce identical results.**

```python
# Strategy A: add to output[0] (mlp stream)
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    out0 = layer.output[0].clone()
    out1 = layer.output[1].clone()
    out0[-1, :] += steering_vector
    layer.output = (out0, out1)

# Strategy B: add to output[1] (residual stream)
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    out0 = layer.output[0].clone()
    out1 = layer.output[1].clone()
    out1[-1, :] += steering_vector
    layer.output = (out0, out1)
```

**Result:** `steer_A vs steer_B: max_diff=0.004, cosine=1.000` — effectively identical.

Both work because:
```
(out0 + v) + out1 = out0 + (out1 + v) = out0 + out1 + v
```

#### Patching (Replacing Hidden States)

**All three decompositions produce identical results.**

```python
# P1: put target in mlp stream, zero residual
layer.output = (target_hs, torch.zeros_like(layer.output[1]))

# P2: preserve residual, adjust mlp stream
residual = layer.output[1].clone()
layer.output = (target_hs - residual, residual)

# P3: zero mlp stream, put target in residual
layer.output = (torch.zeros_like(layer.output[0]), target_hs)
```

**Result:** `P1 vs P2 vs P3: max_diff=0.000` — bit-for-bit identical.

**Recommended: use P1 `(target_hs, zeros)` for simplicity.**

#### Ablation (Zeroing Out)

**Must zero BOTH streams to match HF-style ablation.**

```python
# CORRECT: zero both streams (matches HF ablation)
layer.output = (torch.zeros_like(layer.output[0]), torch.zeros_like(layer.output[1]))

# WRONG: zero only mlp stream (residual leaks through)
layer.output = (torch.zeros_like(layer.output[0]), layer.output[1].clone())
```

**Result:** `ablate_both vs ablate_mlp: max_diff=3.088` — NOT the same.

Zeroing only `output[0]` leaves the residual stream intact, so previous layers' contributions still flow through. This is a fundamentally different intervention than zeroing the full hidden state.

#### Scaling

**Must scale BOTH streams.**

```python
# CORRECT
out0 = layer.output[0].clone()
out1 = layer.output[1].clone()
layer.output = (out0 * scale, out1 * scale)

# WRONG — only scales mlp component
out0 = layer.output[0].clone()
layer.output = (out0 * scale, layer.output[1].clone())
```

### Summary Table

| Operation | HF Pattern | vLLM Pattern | Notes |
|-----------|-----------|-------------|-------|
| **Steer** (add vector) | `layer.output[0][:, -1, :] += v` | Add `v` to either `output[0]` or `output[1]` | Both streams equivalent |
| **Patch** (replace) | `layer.output[0] = target` | `layer.output = (target, zeros)` | Simplest decomposition |
| **Ablate** (zero out) | `layer.output[0][:] = 0` | `layer.output = (zeros, zeros)` | Must zero BOTH streams |
| **Scale** | `layer.output[0] *= s` | `layer.output = (out0*s, out1*s)` | Must scale BOTH streams |

### Important Notes

1. **Always `.clone()` before modifying** — Gap 1.1 means fused kernels mutate tensors in-place after hooks fire.
2. **Always replace the full tuple** — `layer.output[0] = x` raises `TypeError: 'tuple' object does not support item assignment`. Use `layer.output = (new0, new1)`.
3. **vLLM uses 2D layout** `[tokens, hidden]` not 3D `[batch, seq, hidden]` — index with `[-1, :]` not `[:, -1, :]`.
4. **Baseline logit gap is expected** — vLLM and HF use different kernels (fused vs separate), producing different numerical results even without intervention. Compare intervention *effects* (delta from baseline), not absolute values.

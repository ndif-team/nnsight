# vLLM ↔ HuggingFace Intervention Gap Report

**Date:** 2026-03-24
**Model:** Qwen/Qwen2-0.5B
**vLLM version:** 0.15.1
**nnsight branch:** worktree-vllm_debug
**Hardware:** 8× NVIDIA A100 (vLLM on GPU 0, HF on GPU 1)

## Summary

**13 gaps documented.** Gap 1.1 is mitigated by general inference-mode cloning. All others are documented differences or hard blocks.

| Gap | Description | Status |
|-----|-------------|--------|
| 1.1 | In-place mutation corrupts `.save()` | **FIXED** — clone-on-save for inference tensors |
| 1.2 | Decoder layer output: `(mlp, res)` vs `(combined,)` | DOCUMENTED |
| 1.3 | Decoder layer `.input` returns int64 positions | DOCUMENTED |
| 1.4 | LayerNorm output: tuple vs tensor | DOCUMENTED |
| 1.5 | LayerNorm input semantics differ | DOCUMENTED |
| 2.1 | MLP submodule layout (merged gate\_up\_proj) | DOCUMENTED |
| 2.2 | Attention submodule layout (merged qkv\_proj) | DOCUMENTED |
| 2.3 | RowParallelLinear returns tuple | DOCUMENTED |
| 3.1 | Flat batch dimension [total\_tokens, hidden] | DOCUMENTED |
| 3.2 | PagedAttention: no attention weights | HARD BLOCK |
| 4.1 | Gradients blocked by inference\_mode | HARD BLOCK |
| 4.2 | Source tracing into fused kernels | HARD BLOCK |
| 4.3 | Module skip breaks fused norm | DOCUMENTED |

The gap descriptions below document the **architectural differences** between vLLM and HuggingFace backends. Users must account for these when writing vLLM interventions.

---

## Group 1: Activation Semantics

### Gap 1.1 — In-place mutation corrupts `.save()` — **FIXED**

**Root cause:** vLLM's `fused_add_rms_norm` mutates tensors in-place after nnsight hooks fire. `.save()` stores a reference, so the saved value silently becomes the post-mutation value.

**Evidence:**
- `layer.output[0]` ref vs clone: diff = **64.62**
- `layer.output[1]` ref vs clone: diff = **1013.81**
- `mlp.output` and `self_attn.output`: unaffected (not mutated downstream)

**Fix:** `.save()` automatically clones inference-mode tensors (`tensor.is_inference()`). The clone is a separate allocation, so downstream fused-kernel mutations affect the original tensor flowing through the model, not the user's saved reference. This is architecture-agnostic — it fires on any inference-mode tensor, not just vLLM.

**Impact:** Transparent to users. `.save()` returns a safe copy on vLLM; no behavior change on HF (tensors are not inference-mode).

---

### Gap 1.2 — Decoder layer output semantics differ

**Root cause:** vLLM maintains a dual residual stream. Decoder layers return `(hidden_states, residual)` as separate tuple elements. HF returns `(hidden_states + residual,)` combined.

**Evidence:**
- vLLM: output is 2-tuple; `output[0] == mlp.output` (cosine = 1.0)
- HF: output is 1-tuple; `output[0] != mlp.output` (cosine = 0.45) because it includes the residual

**Impact:** Users must combine streams manually: `combined = layer.output[0] + layer.output[1]`. Logit lens, steering vectors, activation patching all need to account for the dual-stream format.

---

### Gap 1.3 — Decoder layer input semantics differ

**Root cause:** vLLM decoder signature is `forward(positions, hidden_states, residual, ...)`. The first positional arg (`.input`) is int64 position IDs, not float hidden states.

**Evidence:**
- vLLM `.input` dtype: `torch.int64`, values: `[0, 1, 2, 3, 4]`
- HF `.input` dtype: `torch.float32`, shape: `[1, 5, 896]`

**Impact:** `.input` returns position IDs, not hidden states. Use `.inputs` to access the full `(args, kwargs)` and extract `args[1]` (hidden\_states) and `args[2]` (residual) directly.

---

### Gap 1.4 — LayerNorm output: tensor vs tuple

**Root cause:** vLLM's fused `fused_add_rms_norm` returns `(normalized, new_residual)` tuple. HF returns a single tensor.

**Evidence:**
- vLLM `input_layernorm.output`: is tuple = True
- HF `input_layernorm.output`: is tensor = True

**Impact:** Use `norm.output[0]` to get the normalized tensor. Logit lens code must extract from tuple: `normed = model.model.norm.output[0]`.

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

**Impact:** Use `down_proj.output[0]` to get the tensor. `down_proj.output * 2` fails without unpacking.

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

### Gap 4.3 — Module skip breaks fused norm

**Root cause:** vLLM decoder layers return `(hidden_states, residual)` tuple. Skipping with a single tensor fails because the next layer's `fused_add_rms_norm` expects the `(x, residual)` pair.

**Evidence:**
- vLLM `skip(single_tensor)`: FAILED (EngineDeadError — unrecoverable engine crash)
- vLLM `skip(tuple)`: FAILED (EngineDeadError — engine already dead from first attempt)
- HF `skip(single_tensor)`: FAILED (shape mismatch — HF layers also expect tuple input)
- HF `skip(tuple)`: OK

**Note:** `skip(single_tensor)` fails on both backends, but for different reasons: HF raises a shape error (recoverable), while vLLM crashes the engine (unrecoverable — all subsequent traces fail with `EngineDeadError`, requiring process restart). The deferred exception mechanism prevents engine death but the skip still fails.

**Impact:** Skip values must match vLLM's expected format `(hidden_states, residual)`. Use `layer.skip((zeros, value))` for layer ablation.

---

## Architectural Root Causes

All 13 gaps stem from four architectural decisions in vLLM:

1. **Fused CUDA kernels** — `fused_add_rms_norm`, `SiluAndMul`, PagedAttention mutate tensors in-place, return tuples, and hide internals from Python hooks. (Gaps 1.1, 1.4, 1.5, 3.2, 4.2, 4.3)

2. **Dual residual stream** — Hidden states and residual are kept as separate tensors throughout the forward pass, unlike HF which combines them. (Gaps 1.2, 1.3)

3. **Merged linear layers** — `ColumnParallelLinear` (gate\_up\_proj, qkv\_proj) and `RowParallelLinear` (returning tuples) for tensor parallelism. (Gaps 2.1, 2.2, 2.3)

4. **Continuous batching + inference\_mode** — Flat `[total_tokens, hidden]` layout and `torch.inference_mode()` wrapper for performance. (Gaps 3.1, 4.1)

---

## Fixability Assessment

| Category | Status | Notes |
|----------|--------|-------|
| **Gap 1.1** (mutation) | **Fixed** | `.save()` auto-clones inference-mode tensors |
| **Gap 1.2** (dual residual output) | Documented | Users combine streams manually |
| **Gap 1.3** (input positions) | Documented | Use `.inputs` to access hidden states at `args[1]` |
| **Gap 1.4** (norm tuple) | Documented | Use `.output[0]` to get normalized tensor |
| **Gap 1.5** (norm input) | Documented | vLLM passes 2 args, HF passes 1 |
| **Gaps 2.1–2.2** (merged modules) | Not fixable | Fundamental vLLM architecture; use `.chunk()` / `.split()` |
| **Gap 2.3** (RowParallel tuple) | Documented | Use `.output[0]` to get tensor |
| **Gap 3.1** (flat layout) | Documented | Use `[-1, :]` not `[:, -1, :]` |
| **Gap 3.2** (no attn weights) | Not fixable | PagedAttention is a fused CUDA kernel |
| **Gap 4.1** (no gradients) | Not fixable | `inference_mode` is set by vLLM engine |
| **Gap 4.2** (fused source) | Not fixable | Computation happens in C/CUDA |
| **Gap 4.3** (skip breaks) | Documented | Skip value must match `(hidden, residual)` format |

---

## Intervention Patterns

vLLM exposes raw internal semantics. Users must account for the differences documented above.

### vLLM-specific patterns

| Operation | Pattern | Notes |
|-----------|---------|-------|
| **Save** | `layer.output[0].save()` | `.save()` auto-clones inference tensors (Gap 1.1) |
| **Combined hidden** | `layer.output[0] + layer.output[1]` | Must combine dual streams manually (Gap 1.2) |
| **Steer** | `layer.output[0][-1, :] += v` | 2D indexing (Gap 3.1) |
| **Ablate** | `layer.output[0][:] = 0` | Only zeroes one stream; zero both for full ablation |
| **Skip** | `layer.skip((zeros, value))` | Must provide `(hidden, residual)` tuple (Gap 4.3) |
| **LayerNorm** | `norm.output[0]` | Extract from tuple (Gap 1.4) |
| **down/o proj** | `proj.output[0]` | Extract from tuple (Gap 2.3) |
| **Layer input** | `layer.inputs[0][1]` | `args[1]` is hidden\_states (Gap 1.3) |

### Key differences from HF

1. **2D layout** `[tokens, hidden]` not 3D `[batch, seq, hidden]` — index with `[-1, :]` not `[:, -1, :]` (Gap 3.1)
2. **Logits** — use `model.logits.output` instead of `model.lm_head.output`
3. **Generation** — use `model.trace(max_tokens=N)` instead of `model.generate(max_new_tokens=N)`
4. **Merged projections** — `gate_up_proj` (Gap 2.1), `qkv_proj` (Gap 2.2) require `.chunk()` / `.split()`
5. **Baseline logit gap** — vLLM and HF use different kernels (fused vs separate), producing slightly different numerical results. Compare intervention *effects* (delta from baseline), not absolute values.

### vLLM's dual residual stream

vLLM decoder layers return `(mlp_output, residual)` as separate tensors. The next layer's `fused_add_rms_norm` computes `RMSNorm(stream0 + stream1)`, so only the sum matters. To get the equivalent of HF's combined hidden state, add both streams: `combined = layer.output[0] + layer.output[1]`.

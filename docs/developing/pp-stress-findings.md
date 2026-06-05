---
title: PP=2 vLLM robustness — intervention stress findings
one_liner: Empirical stress test of the vLLM pipeline-parallel path with real intervention patterns at PP=2; two PP correctness/liveness bugs found (logits-consume-in-iter hang; cross-stage tuple-output read returns wrong values), plus one pre-existing non-PP limitation.
tags: [internals, dev, vllm, pp, testing, bug]
related: [docs/developing/pp-pipeline-parallelism.md, docs/developing/vllm-integration.md]
status: P1 + P2 FIXED & verified at PP=2 across Qwen2/Qwen3/Llama; N1 is a pre-existing non-PP limitation (unchanged)
---

# PP=2 vLLM robustness — intervention stress findings

> **What this is.** A from-scratch stress test of the single-node multiproc **PP=2** path
> (`pipeline_parallel_size=2`, `tensor_parallel_size=1`) on `pp-on-dev`, driving the standard
> interpretability intervention patterns (logit-lens / read-all-layers, activation patching,
> ablation, steering, cross-stage read/write, multi-token generation, concurrent batching) and
> comparing every result against a **PP=1 reference** (same code, one GPU). vLLM here is **0.19.1**
> (the PP path was originally developed against 0.15.1). 4-node Ray PP is a known upstream bug and is
> out of scope — see [pp-multinode-ray-init-bug.md](pp-multinode-ray-init-bug.md).
>
> **Status: documented, not fixed.** Per the request that drove this, no PP logic was changed.

## Summary

| ID | Kind | Pattern that triggers it | Severity | Confirmed on |
|----|------|--------------------------|----------|--------------|
| **P1** | PP liveness (hang) | multi-token `for … tracer.iter[:K]:` loop that **consumes** `model.logits` per step (e.g. `.argmax()`/`.item()`) | High (deadlocks the most natural token-collection idiom) | GPT-2, Qwen2.5-0.5B, **Qwen3-0.6B**, Llama (DeepSeek-R1-Distill-Llama-8B) |
| **P2** | PP correctness (silent) | reading an **upstream** decoder-layer **tuple** output `layers[i].output[0]` and consuming/saving it | High (silently wrong values, no error) | Qwen2.5-0.5B, **Qwen3-0.6B**, **Llama (DeepSeek-8B)** |
| **N1** | non-PP baseline | in-place `module.output[:] = …` write on a vLLM inference tensor | Medium (pre-existing, not PP) | GPT-2, Qwen2.5-0.5B |

What is **robust** at PP=2 (all PASS, PP==PP1): plain generation that appends **raw** `model.logits`
(argmax outside the trace); single-forward cross-stage **read+consume** of a downstream layer
(`cross_consume`); per-step consume of a downstream **layer** output in a loop (`gen_consume_layer`);
cross-stage **replacement** write `module.output = value` (`cross_write`); **ablation**; **steering**;
and **concurrent** multi-prompt multi-token generation (`concurrency`, batch-group isolation intact at
**6 and 16 concurrent invokes**). The PP path works for a wide range of interventions; the two bugs below
are specific.

## Methodology

- Harness: `/tmp/pp_stress/{worker.py,run.py}` (reusable). `worker.py` loads one `VLLM(model,
  tensor_parallel_size=1[, pipeline_parallel_size=2], gpu_memory_utilization=…, dispatch=True)` and runs
  one scenario; `run.py` runs each scenario at PP=1 and PP=2 in separate subprocesses and auto-classifies:
  PASS / PP_HANG / PP_BUG / NONPP_BUG / BASELINE_Q.
- Oracle: greedy (`temperature=0, top_p=1`) so token sequences are deterministic; compare token ids
  exactly, or scalar/norm rel-diff < 2%.
- All layer indices are **derived from the model config at runtime** (early=n//4, mid=n//2, late=3n//4),
  never hardcoded, so the same scenarios run on GPT-2 / Qwen / Llama regardless of depth or naming.
- Single-node multiproc, 2 GPUs. The box is shared/volatile; runs use conservative
  `gpu_memory_utilization` and `HF_HUB_OFFLINE=1`.

---

## P1 — `tracer.iter` loop that consumes `model.logits` per step hangs (readiness-gate timeout)

**Symptom.** A plain multi-token generation that collects tokens the natural way —

```python
with model.trace(prompt, temperature=0.0, max_tokens=K) as tracer:
    ids = list().save()
    for _ in tracer.iter[:K]:
        ids.append(model.logits.argmax(dim=-1))   # <-- consumes logits each step
```

deadlocks at PP=2. The stage-0 worker raises, on the **second** step's forward:

```
TimeoutError: PP readiness gate: mediator … not ahead of forward (worker_iteration=0, k=1) within 30s
  src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:722  (_pp_wait_for_mediators)
```

PP=1 with identical code is fine.

**Trigger pinned (controls, GPT-2 + Qwen2.5-0.5B, ample GPU):**

| pattern | result |
|---|---|
| `with tracer.iter[:K]:` + append **raw** `model.logits` | PASS |
| `for _ in tracer.iter[:K]:` + append **raw** `model.logits` | PASS |
| **single** forward + consume downstream lazies (`read_all` `.norm()` over 24 layers) | PASS |
| `for _ in tracer.iter[:K]:` + consume a downstream **layer** output per step (`.sum()`) | PASS |
| `for _ in tracer.iter[:K]:` + consume **`model.logits`** per step (`.argmax()`) | **HANG** |

So it is **not** the model, **not** the `for`-vs-`with` form, and **not** generic downstream-consume.
It is specifically **consuming the terminal `model.logits` (last-stage eproperty) per step inside the
iter loop**.

**Root cause (CONFIRMED).** The cross-stage pull for `model.logits` is built with **`source_rank=None`**,
so it is misdirected and its `dist.recv` blocks forever.

- On a non-last stage `model.logits` correctly short-circuits to a `LazyRemoteTensor` — `_is_pp_missing`
  (`pp_envoy.py:131`) resolves `lookup = f"{obj_path}.{key}"` = `"model.logits"` and `pp_module_map`
  reports it remote. (The stale "WrapperModule" theory in `tests/test_vllm_pp_integration.py`'s docstring
  is out of date; logits/samples are handled.)
- But `_pp_lazy_access` (`pp_envoy.py:164`) resolves the pull's owner as
  `source_rank = pp_map.get_owning_rank(path or key)` — and for the root-level `logits`/`samples`
  epropertys `obj.path` is the **VLLM root `"model"`** (the module name lives in `key`, not `path`). So it
  calls `get_owning_rank("model")`, which matches no layer-container and no first/last-rank name →
  **returns `None`** (`pp.py:85-115`). Verified directly: `get_owning_rank("model")=None`,
  `get_owning_rank("model.logits")=1` (last rank), `get_owning_rank("model.model.layers.6")=0`.
- `.argmax()` materializes the lazy (`lazy_remote_tensor.py:40`) → `pull_from_remote(source_rank=None, …)`
  → `dist.send(group_dst=None)` then `dist.recv(group_src=None)` waits for a reply that never arrives.
  The stage-0 mediator is stuck inside iteration 0's body, never advances `_pp_worker_iteration` 0→1
  (`iterator.py:88`); the next forward's readiness gate (`GPUModelRunner._pp_wait_for_mediators`,
  `_ahead(m,k)` at `GPUModelRunner.py:702`) waits for `worker_iteration >= k=1`, never sees it, times out.

**Why it's logits/samples-specific:** decoder layers are accessed on **sub-envoys** whose `.path` *is* the
layer path (`"model.model.layers.6"`), which `get_owning_rank` resolves via the layer container → correct
rank; so `gen_consume_layer` works. logits/samples are epropertys on the **root** envoy → `.path="model"`
→ `None`. **Why only consume-in-loop:** raw `.append(model.logits)` and single-forward `.save()` never
materialize (no pull); only a real consume triggers `_materialize` → the misdirected pull.

**The inconsistency in one line:** `_is_pp_missing` uses `f"{path}.{key}"` (`"model.logits"`) for
*detection*, but `_pp_lazy_access` uses `path or key` (`"model"`) for *owner resolution*. They must use
the same lookup.

**Fix (APPLIED + VERIFIED).** In `_pp_lazy_access`, resolve the owner from the full key (already computed
as `module_key = f"{path}.{key}"`):
```python
# pp_envoy.py:164
- source_rank = pp_map.get_owning_rank(path or key)
+ source_rank = pp_map.get_owning_rank(module_key)
```
Plus hardening: added `"logits_processor"` to `_LAST_RANK_MODULES` (`pp.py:55`) so direct access to the
raw processor module on archs that build it on every rank (Qwen2/GPT2/OPT/Pythia/Bloom/Gemma2) also
short-circuits instead of blocking on a hook that never fires.

`get_owning_rank(module_key)` gives logits→last-rank, samples→last-rank, and leaves layers unchanged
(`"…layers.6.output"`→0, same as `"…layers.6"`→0). **Verified at PP=2:** `clean` (logits-consume-in-iter)
now PASSES on Qwen2.5-0.5B and Qwen3-0.6B (pp1==pp2 token sequences) and runs to completion on
DeepSeek-Llama-8B (pp2 valid tokens; was a hang). No regressions across `gen_for`, `gen_consume_layer`,
`read_split_mlp`, `cross_write`, `steering`, `concurrency`. The corrected pull key-matched stage-1's local
logits buffer, so no second bug. (P2 below is unaffected — `read_split` still fails identically.)

**Workaround (no code change):** append **raw** `model.logits` inside the loop and `.argmax()`/`.item()`
**outside** the trace. (The deprecated `with tracer.iter[…]:` form also happens to dodge it.)

**Why it matters:** `for … tracer.iter[…]:` is the recommended idiom and `model.logits.argmax()` /
`model.samples.item()` per step is the obvious way to collect generated tokens — so the most natural
generation-with-logits loop hangs under PP.

---

## P2 — cross-stage read of an upstream decoder-layer **tuple** output returns wrong values (silent)

**Symptom.** Reading early/upstream decoder-layer outputs and consuming them yields **wrong numbers, no
error**. In a 24-layer Qwen2.5-0.5B at PP=2 (stage 0 = layers 0–11, stage 1 = 12–23), reading every
layer's residual norm:

```
L0–L10  (stage-0 internal layers): pp1 ≠ pp2   ← WRONG at PP=2 (e.g. L6: 20.05 → 115.3)
L11     (stage-0 boundary layer):  pp1 = pp2   ← correct
L12–L23 (stage-1 layers):          pp1 = pp2   ← correct
```

**Isolation.**
- Separate scalar saves (no list, removing any save-merge ambiguity): early layer still **wrong**
  (L6 20.05→115.3), late layer **correct** ⇒ not a list-merge artifact; it's the cross-stage read.
- **Tuple-specific (decisive):** same layer L6, same forward — reading `layers[6].output[0]`
  (decoder-layer **tuple** element) is **wrong**, but reading `layers[6].mlp.output` (**single tensor**)
  is **correct** (20.05 == 20.05). `cross_write` (consumes an early MLP output across stages) is also
  correct. So single-tensor submodule outputs pull fine; **tuple-valued decoder-layer outputs do not.**
- The stage-0 **boundary** layer (L11) is correct because its output is the genuine inter-stage
  activation that vLLM actually transfers to stage 1; the internal layers are only buffered, and that
  buffered pull is wrong.

**Same tensor, two read paths — only the tuple path is wrong.** In vLLM's fused-residual Llama/Qwen
decoder layer, `forward` returns `(hidden_states, residual)` where the returned `hidden_states` *is the
MLP output* (the residual stream is the second element). So `layers[i].output[0]` and
`layers[i].mlp.output` are the **same underlying tensor** — and at PP=1 they have identical norms
(Qwen2.5-0.5B early=20.053 both; Qwen3-0.6B early=53.372 both). At **PP=2** the tuple path
`layers[i].output[0]` returns a wrong value (Qwen3 86.76 vs the correct 53.37) while `mlp.output`
returns the correct one. That isolates the fault to **tuple-element handling in the cross-stage pull**,
not the value itself. The same same-tensor check confirms it on **Llama-8B** from pp2-only data
(an upstream layer: `output[0]`=91.9 vs `mlp.output`=11.1 — must be equal, so the tuple read is wrong;
a stage-1-local layer: `output[0]`=`mlp.output`=24.4 — equal/correct, so **only the upstream
cross-stage tuple pull is broken, not local tuple reads**).

**Mechanism (from source).** `layers[i].output[0]` is `lazy[0]` → a child lazy whose deferred pull does
`parent._materialize()[0]` (`lazy_remote_tensor.py:117`). The parent pull returns the buffered upstream
layer output from the source rank; for a **tuple-valued** output the pulled element-0 is wrong (its
magnitude resembles the **residual stream**, i.e. element 1).

**Root cause (diagnosed).** The per-rank `pp_hook_buffer` clone only deep-copies a **bare tensor**;
tuple/list module outputs are stored **by reference, un-cloned** (`interleaver.py:1287-1291`):
```python
stored = value.clone() if isinstance(value, torch.Tensor) else value   # tuple → no clone
```
vLLM runs eager with aggressive in-place / buffer-reuse in its fused add-RMSNorm, so a decoder layer's
`(hidden, residual)` tensors are overwritten by **subsequent** layers before the cross-stage pull is
served. The transfer/recv path is sound — `_serve_reply` (`pp_listener.py:347-379`) and `_recv_legacy`
(`pp_listener.py:495-536`) correctly send/rebuild a multi-tensor tuple in order — so the wrong value comes
purely from the buffer holding stale tensors at serve time. This exactly predicts the observed pattern:
the **last** stage-0 layer (boundary) is correct because nothing after it mutates its buffer, while every
**internal** upstream layer is wrong; and single-tensor `mlp.output` is correct because the `Tensor`
branch *does* clone. Same family as the clone-on-save boundary issue
([project_pp_boundary_capture_leak]/PR #662), but on the PP cross-stage buffer.

**Fix (APPLIED + VERIFIED).** Deep-clone the buffered value's tensors, not just bare tensors — added
`_deep_clone` (`interleaver.py`) and used it at the single buffer-write site (`interleaver.py:1289`):
```python
def _deep_clone(v):
    if isinstance(v, torch.Tensor): return v.clone()
    if isinstance(v, (tuple, list)): return type(v)(_deep_clone(x) for x in v)
    if isinstance(v, dict): return {k: _deep_clone(x) for k, x in v.items()}
    return v
stored = _deep_clone(value)   # under torch.inference_mode(), as today
```
Both serve paths use the cloned `stored` (the recv-loop reads the buffer dict; `dispatch_parked` is
passed `stored`), so one site covers everything. **Verified at PP=2:** `read_split` PASSES on
Qwen2.5-0.5B (early 20.05==20.05) and Llama-8B (early 11.14==11.14, was 91.9); `read_all` matches PP=1
with **max reldiff 0.0 across all layers** on Qwen3-0.6B (28) and Llama-8B (32). No regressions. A second factor amplifies it: consuming via `.float().norm()` **eagerly materializes** the
wrong pull into a real scalar on the non-owning rank, so `strip_lazy` can't sentinel it and
`merge_saved`'s "b-wins" rule (`lazy_remote_tensor.py:295`) surfaces that wrong value over the owning
rank's correct local one. Fixing the tuple pull (factor 1) is the root fix; the merge preference is
secondary.

**Why it matters:** breaks **logit-lens / "read every layer"** and **"capture an early-layer
activation"** workloads under PP — and `layers[i].output[0]` is exactly the form used in the PP design
doc's own running example. This is the dangerous kind of bug: silent, plausible-looking wrong numbers.

---

## Non-PP findings (documented separately, as requested)

### N1 — in-place `module.output[:] = …` errors on vLLM inference tensors (baseline, not PP)

`model.<…>.output[0][:] = x` / `module.output[:] = 0` raises at **PP=1** (and PP=2 owning rank):

```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```

Reproduced on GPT-2 and Qwen2.5-0.5B at PP=1 — so it is a **baseline vLLM-path limitation**, not PP
(matches the in-place note in the canonical vLLM doc / prior work). The supported idiom is
**replacement** (`module.output = value`), which works at both PP=1 and PP=2 (`cross_write`, `ablation`,
`steering` all PASS). On GPT-2, the in-place write happened to succeed at PP=2 because the write target
was a no-op `LazyRemoteTensor` on the non-owning rank — i.e. PP can *mask* this baseline error, which is
itself a reason to prefer the replacement form.

## Not bugs — harness/usage caveats (recorded so they aren't miscounted)

- **`tracer.cache()`** is documented as **not validated on the vLLM path** (vllm.md Limitations), and the
  cache API is attribute-navigation (`cache.model.layers[i].output`), with entries accumulating across
  forwards — the initial harness misuse (`cache[key].output`) produced a false "bug". Out of scope until
  cache-on-vLLM is targeted.
- **Cross-prompt activation patching** (`tracer.barrier(2)` + two invokes, capture in A / patch in B):
  the first harness version failed at **PP=1 too** (an `UnboundLocalError` from a `.save()` var assigned
  inside a nested `with tracer.invoke()`), so it is not a PP bug; it needs a corrected harness structure
  to retest and is deferred.

## Environment caveats

- **Gated Llama tokenizers:** `meta-llama/*` repos are gated and the local cache here is weights-only
  (no tokenizer) with no HF token → tokenizer load 401s (online) / `TypeError: not a string` (offline).
  Use a non-gated Llama-architecture model (e.g. `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`) or supply a
  tokenizer. Set `HF_HUB_OFFLINE=1` for hermetic local-cache runs.
- **Shared GPUs:** other users make free memory volatile; a too-high `gpu_memory_utilization` yields
  `ValueError: Free memory … less than desired …` at startup — environmental, not a bug. Pin to the
  most-free GPUs and keep util conservative.

## Reproduce

```bash
PY=/disk/u/zikai/anaconda3/envs/ndif-dev/bin/python   # vllm 0.19.1
cd /tmp/pp_stress
# P1 (hang) + P2 (read_split wrong) + controls + passing scenarios, PP1 vs PP2:
$PY -u run.py --model Qwen/Qwen2.5-0.5B-Instruct --util 0.2 --timeout 100 \
  --scenarios clean,gen_for,gen_consume_layer,read_split,read_split_mlp,read_all,\
cross_consume,cross_write,ablation,steering,concurrency,inplace
```

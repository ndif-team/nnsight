# NNsight Workload Analysis: From an Inference Engine's Perspective

This document summarizes an analysis of nnsight's workloads, batching patterns, and execution overhead from the perspective of an inference engine designer.

---

## 1. Workload Profile: Prefill-Dominant

NNsight workloads are overwhelmingly **prefill-only** (single forward pass).

| API | Usage Share | Inference Pattern |
|---|---|---|
| `model.trace()` | ~82% | Single forward pass (prefill only) |
| `model.generate()` | ~18% | Prefill + N decode steps (typically N=3-10) |

**Implications:**
- **Request-level KV cache** (for autoregressive decode) provides near-zero benefit for 82% of workloads. The 18% generation workloads are typically short.
- **Prefix caching** is not straightforwardly beneficial. Standard prefix caching assumes `same tokens → same KV`, but nnsight interventions violate this invariant — they modify activations while keeping input tokens identical, producing different KV states. Prefix caching would return incorrect results for intervention workloads without intervention-aware invalidation.

---

## 2. Batching Architecture: Invokes and Mediators

### How Batching Works

NNsight batches multiple inputs via the **invoke** pattern within a single `trace()`. Each invoke creates a **mediator** (a worker thread), but they all share one forward pass.

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):     # Mediator 1, batch_group=[0,1]
        h1 = model.layer5.output.save()
    with tracer.invoke("World"):     # Mediator 2, batch_group=[1,1]
        h2 = model.layer5.output.save()
```

**The engine sees:** one `model(input_ids=[2, seq_len], attention_mask=...)` call — standard HuggingFace padded-batch prefill. The mediator threading is invisible to the engine.

### Batching Pipeline

```
invoke("Hello") → _prepare_input() → ids=[1,2]      batch_group=[0,1]
invoke("World") → _prepare_input() → ids=[3,4,5]    batch_group=[1,1]
                        │
                   _batch() pads:
                   ids=[[1,2,0],[3,4,5]]  mask=[[1,1,0],[1,1,1]]
                        │
              ONE forward pass with combined batch
                        │
              At each hook: narrow() extracts each invoke's slice
              Mediator 1 sees output[0:1], Mediator 2 sees output[1:2]
```

### Mediator Protocol: Strictly Serial

Each mediator runs in a separate OS thread, but the main thread and worker threads **alternate in lock-step** using `_thread.lock`:

```
Main thread          Worker thread
     │                     │
     │   respond(value)    │
     ├───put(value)───────►│  worker wakes up
     │                     │  runs user code
     │   BLOCKED           │  ...
     │   (lock.acquire)    │  ...
     │                     │  hits next .output access
     │◄──put(event)────────┤  sends event → releases main
     │                     │  BLOCKED (response_queue.wait)
     │   process event     │
     ▼                     │
  next mediator            │  still blocked
```

**Zero parallelism between workers.** The `_thread.lock` protocol enforces strict alternation — exactly one thread runs at a time.

---

## 3. Batching Benefit for Short Prompts

### Prefill Is NOT Always Compute-Bound

Whether prefill is compute-bound or memory-bandwidth-bound depends on `B × S` (batch_size × sequence_length):

```
Arithmetic intensity = B × S   (ops/byte, FP16)

A100 threshold: 312 TFLOPS / 2 TB/s = 156 ops/byte

B × S > 156 → compute-bound (GPU arithmetic is bottleneck)
B × S < 156 → bandwidth-bound (weight loading is bottleneck)
```

Most interpretability prompts are short (5-20 tokens). At B=1, S=7: B×S=7 — **heavily bandwidth-bound** (22× below threshold).

### Batching Throughput (Llama-8B, A100, S=7)

| Batch Size | B×S | Regime | Wall Time | Throughput |
|---|---|---|---|---|
| 1 | 7 | BW-bound | ~7 ms | 143 req/s |
| 8 | 56 | BW-bound | ~7 ms | 1,143 req/s |
| 23 | 161 | Crossover | ~7 ms | 3,286 req/s |
| 32 | 224 | Compute-bound | ~10.5 ms | 3,048 req/s |

**Up to B≈23, batching is essentially free** — the GPU was idle waiting for weight loads anyway. Each additional request adds negligible marginal cost until the compute-bound threshold.

### Empirical: Raw Batching vs NNsight Batching (Scenario 3)

The paper analysis above describes raw model batching (no hooks). In practice, nnsight batching adds a mediator per invoke, so each additional request increases both GPU batch size **and** hook overhead. The benchmark compares raw `model(**tokenized_batch)` vs `model.trace()` with B invokes, each doing a single `.save()`.

**Qwen2.5-7B, A100, S≈7:**

| B | Raw batched | NNsight batched | Overhead ratio | Abs gap | Raw thr | NN thr |
|---|---|---|---|---|---|---|
| 1 | 37.5 ms | 49.0 ms | 1.30× | +11.5 ms | 27 req/s | 20 req/s |
| 2 | 36.9 ms | 49.8 ms | 1.35× | +12.9 ms | 54 req/s | 40 req/s |
| 4 | 36.5 ms | 53.3 ms | 1.46× | +16.8 ms | 110 req/s | 75 req/s |
| 8 | 29.1 ms | 50.3 ms | 1.73× | +21.2 ms | 275 req/s | 159 req/s |
| 16 | 35.9 ms | 68.9 ms | 1.92× | +33.0 ms | 445 req/s | 232 req/s |
| 32 | 45.9 ms | 97.2 ms | **2.12×** | +51.3 ms | 697 req/s | 329 req/s |

**Key observations:**

1. **Overhead ratio grows with B.** At B=1, nnsight adds 1.30× overhead. At B=32, it's 2.12×. Each invoke adds a mediator, so hook iteration cost scales linearly with B — but raw batching is nearly free in the bandwidth-bound regime, so the raw baseline stays flat while nnsight's cost climbs.

2. **The absolute gap roughly doubles when B doubles** (+11.5 ms at B=1, +51.3 ms at B=32). This is consistent with Scenario 2's measured 1.96 ms/mediator slope: at B=32 we'd predict ~63 ms of mediator overhead, and the measured gap is 51 ms — in the right ballpark.

3. **Throughput diverges.** Raw batching reaches 697 req/s at B=32 (still bandwidth-bound, still scaling). NNsight tops out at 329 req/s — **less than half** — because the serial mediator overhead becomes the bottleneck long before GPU compute does.

4. **Batching still helps within nnsight** — 329 req/s batched vs ~20 req/s sequential is a large win. The point is that it helps *less* than it should compared to raw batching, and the gap widens with batch size.

> **Gap to investigate:** The raw batched time fluctuates (29-46 ms across B values) rather than monotonically increasing. This may be due to padding inefficiency (different prompt lengths), CUDA kernel launch variability, or the tokenizer's padding strategy. A controlled experiment with fixed-length prompts would give a cleaner scaling curve. Also, the paper predicts batching is "free" up to B≈23, but the raw benchmark shows near-constant time only up to B≈8 before starting to increase — the compute-bound crossover may be lower than the theoretical B×S=156 threshold suggests, possibly due to attention's quadratic cost or memory bandwidth contention from padding.

---

## 4. Hook Overhead: The Dominant Cost

### Hook Architecture

NNsight registers PyTorch forward hooks (input + output) on **every module** in the model via the Envoy wrapping system. These hooks fire during every forward pass and iterate through all alive mediators.

For Llama-8B: ~421 modules → ~842 hook calls per forward pass.

### Hooks Are Fully Blocking

Hooks **must** be synchronous — they can modify outputs before the next module sees them. The next module's GPU kernels cannot launch until the hook returns.

```python
# Module.__call__ (simplified)
def __call__(self, *args, **kwargs):
    args, kwargs = input_hook(self, args, kwargs)    # CPU blocks
    output = self.forward(*args, **kwargs)           # launches GPU kernel
    output = output_hook(self, args, output)         # CPU blocks
    return output
```

### Per-Hook Cost

| Scenario | Cost per mediator | CUDA sync? |
|---|---|---|
| No-match (99.8% of hooks) | ~1 μs | No (string/queue ops only) |
| Match, `.save()` only | ~15 μs (thread switch + save) | No |
| Match, read tensor values | ~(Y + 25) μs | Yes — drains GPU pipeline |
| Match, in-place modify | ~20 μs | No (launches kernel async) |

### The Wrapper Module Problem

Models contain **wrapper modules** (LlamaDecoderLayer, LlamaAttention, LlamaMLP) that don't launch GPU kernels — they just call children. But they still have hooks:

- Per layer: 3 wrapper modules with Y=0 GPU time
- Hooks on these are **always** pure overhead — no GPU work to overlap with

### Per-Module Overhead Formula

For a module with GPU kernel time Y, per-hook CPU time X = 2 + M μs (M = alive mediators):

```
Per-module overhead = max(0, 2X + 5 - Y)

Input hook:  X μs  (runs BEFORE kernel — no overlap possible)
Launch:      5 μs
Output hook: X μs  (partial overlap with GPU kernel for no-match, no CUDA sync case)
```

### Overhead Estimates (Llama-8B, A100, S=7) — Paper Analysis

Per-layer overhead (32 layers + top-level):

| M (mediators) | Per-hook X | Per-layer OH | Total OH | Forward pass | Slowdown |
|---|---|---|---|---|---|
| 1 | 3 μs | 85 μs | **2.8 ms** | 7 ms | 1.4× |
| 4 | 6 μs | 141 μs | **4.6 ms** | 7 ms | 1.7× |
| 8 | 10 μs | 229 μs | **7.4 ms** | 7 ms | 2.1× |
| 32 | 34 μs | 735 μs | **23.5 ms** | 10.5 ms (B=32) | 3.2× |

**Even 1 mediator costs ~40% overhead**, primarily from wrapper modules.

### Empirical Benchmark Results (A100 80GB)

Benchmark: `tests/performance/benchmark_batching.py` — raw model forward vs nnsight trace, single `.save()` at a mid layer.

**Hook overhead baseline (Scenario 1):**

| Model | Raw forward | Empty trace | Single save | Overhead (1 mediator) |
|---|---|---|---|---|
| GPT-2 (124M, 164 modules) | 9.2 ms | 12.8 ms | 13.4 ms | **1.45×** |
| Qwen2.5-7B (400+ modules) | 38.3 ms | 47.0 ms | 51.1 ms | **1.33×** |

GPT-2's higher relative overhead (1.45× vs 1.33×) is expected: the hook cost is roughly fixed (~4-12ms) but GPT-2's forward pass is much shorter, so the same absolute cost is a larger fraction. The paper estimate of ~1.4× for 1 mediator is in the right ballpark.

**Mediator scaling (Scenario 2):**

| Model | Slope (ms/mediator) | R² | Predicted overhead at M=16 |
|---|---|---|---|
| GPT-2 | 1.11 | 0.996 | ~18 ms |
| Qwen2.5-7B | 1.96 | 0.982 | ~31 ms |

Scaling is highly linear (R² > 0.98). Qwen's higher per-mediator cost (~1.8× GPT-2) tracks its ~2.4× larger module count, consistent with the O(modules × mediators) model.

> **Gap to investigate:** The paper predicts per-hook cost X = 2 + M μs, giving ~34 μs per hook at M=32 and ~23.5 ms total overhead. The benchmark measures ~1.96 ms/mediator on Qwen, which at M=16 gives ~31 ms overhead on top of ~48 ms base — a 1.65× slowdown. The paper predicted 3.2× at M=32. The discrepancy may come from: (a) the paper assumes all 842 hooks iterate all mediators, but the actual code may short-circuit in some cases; (b) the paper's per-hook μs estimate may be too high; (c) modern Python/PyTorch may have lower per-iteration overhead. This needs profiling at the hook level to resolve.

### Scaling: O(total_modules × alive_mediators)

The overhead scales with the product of total modules and alive mediators. Of ~842 hook calls per forward pass, typically only 1-2 actually match a mediator's request. **~99.8% of hook calls are waste** — they exist to support the possibility of intervention at any module.

---

## 5. CUDA Synchronization at Match Points

When a mediator's intervention reads tensor **values** (not just handles), an implicit CUDA sync occurs:

- `.save()` — stores tensor handle, **no sync**
- `tensor.mean()`, `print(tensor)`, `.item()` — **CUDA sync** (GPU must finish kernel first)
- `tensor[:] = 0` — launches async kernel, **no sync**

At a CUDA sync point, the GPU pipeline drains completely. The GPU has no queued work and must refill from scratch after the sync.

**However, CUDA sync adds relatively little to total overhead** (~5-12% additional) because it only affects the 1-2 match hooks per mediator, while the 842×M no-match iterations dominate:

```
Overhead breakdown for M=32, B=32:

No-match hooks:     23.5 ms  (91%)  ← DOMINANT
Match + CUDA sync:   1.4 ms   (5%)
Pipeline drain:      1.6 ms   (6%)
```

### Empirical CUDA Sync Results (Scenario 5)

| Variant | GPT-2 | Qwen2.5-7B | Description |
|---|---|---|---|
| `save_only` | 14.1 ms | 39.3 ms | Tensor handle only, no sync |
| `value_read` (.mean()) | 14.8 ms (+5%) | 43.2 ms (**+9.8%**) | Forces CUDA sync |
| `in_place_modify` ([:]=0) | 13.1 ms (-7%) | 39.6 ms (+0.8%) | Async kernel, no sync |

On GPT-2, the differences are in the noise. On Qwen2.5-7B, `value_read` adds a measurable **+9.8%** overhead — consistent with the paper's 5-12% estimate. In-place modification has negligible cost because it launches an async kernel without waiting.

> **Gap to investigate:** The paper predicts sync overhead is small relative to hook iteration cost (91% no-match). The empirical +9.8% on Qwen suggests the sync cost may be slightly larger than estimated at M=1, or that pipeline drain is more expensive on larger models. Profiling individual hook call times at match vs no-match points would clarify.

---

## 6. Head-of-Line Blocking: Heavy Interventions Stall the Batch

Since mediators are processed **sequentially** (lock-step threading), one slow mediator blocks the entire batch.

### The Mechanism

When a worker does heavy work (SAE, classifier, iterative optimization) between receiving a value and sending its next event, the main thread is stuck in `event_queue.wait()`:

```
Hook fires at layer 5 (32 mediators, one runs a 5ms SAE):

Main: [med1:15μs][med2:15μs]...[med17: SAE 5ms ████████][med18:15μs]...[med32:15μs]
GPU:  [████████████████████ IDLE ████████████████████████████████████████████████████]
```

### Impact Scenarios (32 requests, S=7, Llama-8B, A100)

| Workload | GPU | No-match OH | Intervention | Total | Batch speedup vs sequential |
|---|---|---|---|---|---|
| No nnsight | 10.5 ms | — | — | **10.5 ms** | 21.3× |
| `.save()` only | 10.5 ms | 23.5 ms | 0.5 ms | **34.5 ms** | 9.1× |
| 1 slow SAE (5ms) | 10.5 ms | 23.5 ms | 6.2 ms | **40.2 ms** | 7.8× |
| All 32 run SAEs | 10.5 ms | 23.5 ms | 182 ms | **216 ms** | 2.3× |
| Iterative steering ×1 | 10.5 ms | 23.5 ms | 70 ms | **104 ms** | 4.8× |

### Time Breakdown for Heavy Workloads

```
All 32 SAEs case (216 ms total):

GPU forward pass:      10.5 ms  █░░░░░░░░░░░░░░░░░  5%
No-match hook OH:      23.5 ms  ██░░░░░░░░░░░░░░░░ 11%
SAE serial execution: 182.0 ms  ████████████████░░ 84%  ← GPU fully idle
```

**The heavier the intervention, the less batching helps.** The batching speedup collapses from 21× toward 1× as intervention cost grows, because:
1. Heavy CPU work serializes across mediators (lock-step protocol)
2. GPU sits completely idle during all CPU-side intervention work
3. One slow mediator penalizes every request in the batch

### Empirical HOL Blocking Results (Scenario 4, B=8)

| Weight | Intervention | Qwen seq (ms) | Qwen bat (ms) | Speedup | GPT-2 speedup |
|---|---|---|---|---|---|
| `save_only` | `.save()` only | 303 | 51 | **5.95×** | 4.48× |
| `light` | `.mean()` + `.save()` | 303 | 54 | **5.56×** | 3.87× |
| `medium` | hidden_dim × hidden_dim matmul + write-back | 305 | 61 | **5.02×** | 3.27× |
| `heavy` | SAE-like: 2× (hidden_dim × 4·hidden_dim) + ReLU | 305 | 63 | **4.80×** | 3.13× |

The trend is clear: **speedup decreases monotonically** from save_only to heavy. On Qwen, it drops from 5.95× to 4.80× (19% reduction). On GPT-2, the drop is steeper: 4.48× to 3.13× (30% reduction) because GPT-2's smaller forward pass makes the intervention cost a larger fraction of total time.

However, the collapse is more modest than the paper predicted (21× → 2.3× for "all 32 SAEs"). The gap comes from several factors:

1. **B=8 vs B=32.** Our benchmark uses B=8 (the paper models B=32). Fewer mediators means less serial stacking of heavy work.
2. **GPU-side intervention.** The matmul interventions (`h @ W`) run on GPU, not CPU. The paper models SAE work as purely CPU-side, but `torch.relu(h @ W_enc)` launches GPU kernels that partially overlap with other work. The serial mediator protocol still applies — each mediator waits for its GPU kernel to finish before releasing the main thread — but the GPU isn't "fully idle" as the paper predicts.
3. **Qwen is expensive relative to interventions.** A 7B forward pass (~38 ms) dwarfs even the heavy intervention (~2.5 ms per invoke). The ratio would be worse on a smaller model or with truly CPU-bound work (Python loops, external API calls).

> **Gap to investigate:** To fully reproduce the paper's predicted collapse, we would need: (a) B=32 with heavy interventions; (b) CPU-bound interventions (not GPU matmuls) to maximize serial blocking; (c) a latency breakdown showing per-mediator stall time during hook execution. The current benchmark's GPU-side heavy intervention may understate the blocking effect because GPU kernels provide some overlap. A variant with `time.sleep()` or Python-loop-based interventions would isolate the serial threading cost.

---

## 7. Key Architectural Observations

### Why the Overhead Exists

1. **Hooks on all modules:** ~842 hooks fire per forward pass, but only 1-2 match. The rest are pure iteration overhead.
2. **Serial mediator processing:** The `_thread.lock`-based protocol enforces strict main↔worker alternation. No parallelism between workers.
3. **Wrapper modules:** Container modules (DecoderLayer, Attention, MLP) have Y=0 GPU time but still have hooks — always pure overhead.

### What Would Reduce Overhead

| Optimization | Potential Reduction | Note |
|---|---|---|
| **Selective hooks** (only on accessed modules) | ~99.8% fewer hook calls | Requires knowing accessed modules at trace compile time |
| **Parallel worker dispatch** (for read-only) | M× speedup at match points | Breaks for write interventions (race conditions on shared tensor) |
| **Mediator indexing** (hash map per provider) | O(1) instead of O(M) per hook | Simple data structure change |
| **CUDA graphs** for non-intervened sections | Near-zero overhead for clean segments | Incompatible with current hook-everywhere design |

### The Fundamental Tension

NNsight's architecture trades **performance for generality**. The hook-on-everything + serial-mediator design allows intervention at any module with any operation, but pays O(modules × mediators) overhead even when only a tiny fraction of modules are accessed. For lightweight read-only workloads this is acceptable (~40% overhead at M=1). For heavy interventions batched together, the serial mediator protocol becomes the dominant cost.

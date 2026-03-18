# Run:AI Model Loading with GPU-Direct Tensor Placement

Fast model loading for nnsight's `LanguageModel` via [run:ai model streamer](https://github.com/run-ai/runai-model-streamer) with direct buffer-to-GPU tensor placement, bypassing CPU intermediaries.

## Loading Paths

### 1. HF `from_pretrained` (baseline)

Standard HuggingFace loading: safetensors files are memory-mapped, tensors are materialized via 4 KB page faults, then HF worker threads call `.to(device)` for GPU placement. Bottlenecked by mmap read granularity and synchronous `cudaMemcpyAsync` blocking on each worker thread.

### 2. Run:AI Stream (CPU clone)

Replace mmap with run:ai's `SafetensorsStreamer` — large sequential `O_DIRECT` reads issued by N concurrent C++ pthreads (no GIL contention). Tensors are cloned to a CPU cache as they arrive, then HF workers call `.to(device)` for GPU placement. Solves the disk I/O bottleneck but CPU→GPU copies still block HF workers.

### 3. Run:AI GPU-Direct (current default)

The loader thread resolves `device_map` before streaming begins, then copies tensors directly from the Run:AI buffer to the target GPU via `.to(device=cuda:N, dtype=dtype)`. When HF workers later call `.to()`, it's a no-op since the tensor is already on the correct device and dtype. The Run:AI streamer's background I/O threads read the next shard from disk concurrently with the blocking `.to()`, providing natural overlap between disk reads and GPU transfers.

```
HF baseline:     disk → mmap page faults → CPU → HF .to(cuda) → GPU
Run:AI stream:   disk → O_DIRECT buffer → clone() → CPU cache → HF .to(cuda) → GPU
Run:AI GPU-direct: disk → O_DIRECT buffer → .to(cuda) → GPU cache → HF .to() [no-op]
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ TransformersModel._load_streamed()                          │
│                                                             │
│  1. resolve_shard_paths()     → list of .safetensors paths  │
│  2. _resolve_device_map()    → {"model.layers.0": 0, ...}  │
│  3. build_lazy_state_dict()  → {key: LazyRunAITensor, ...}  │
│  4. from_pretrained(None, state_dict=...)                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RunAIShardCache (loader thread)                             │
│                                                             │
│  SafetensorsStreamer.get_tensors() yields (name, buffer)    │
│    ├─ GPU target? → tensor.to(device=cuda:N, dtype=dtype)  │
│    └─ CPU target? → tensor.clone()                         │
│  Store in _tensors[name], notify_all() waiting workers     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ HF GLOBAL_WORKERS threads                                   │
│                                                             │
│  _materialize_copy(lazy_tensor, device, dtype)              │
│    → lazy_tensor[...]  → cache.get() → returns GPU tensor  │
│    → tensor.to(device, dtype)  → no-op (already there)     │
└─────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

- **`_resolve_device_map` dtype fix:** The meta model used for device map resolution is created with the model's native dtype (from config) instead of float32 default. Without this, large models (e.g., Qwen3-32B at 61 GB in BF16 vs 122 GB in FP32) get incorrect memory estimates and trigger disk offloading.

- **`expandable_segments`:** GPU-direct streaming allocates hundreds of individually-sized tensors on GPU, which can cause CUDA memory fragmentation. The loader automatically enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on first GPU-direct use.

### Key Files

| File | Role |
|---|---|
| `src/nnsight/modeling/loader.py` | `RunAIShardCache`, `LazyRunAITensor`, `build_lazy_state_dict` |
| `src/nnsight/modeling/transformers.py` | `_load_streamed`, `_resolve_device_map`, `gpu_direct` flag |

### User-Facing API

```python
from nnsight import LanguageModel

# GPU-direct (default when run:ai is installed)
model = LanguageModel("Qwen/Qwen3-8B", device_map="auto", dispatch=True)

# Force HF from_pretrained
model = LanguageModel("Qwen/Qwen3-8B", device_map="auto", dispatch=True,
                      load_format="from_pretrained")

# CPU-clone path (no GPU-direct)
model = LanguageModel("Qwen/Qwen3-8B", device_map="auto", dispatch=True,
                      gpu_direct=False)

# Tune Run:AI I/O concurrency (default 16)
model = LanguageModel("Qwen/Qwen3-8B", device_map="auto", dispatch=True,
                      concurrency=32)
```

If `runai-model-streamer` is not installed, loading falls back to `from_pretrained` silently.

## Benchmark Results

### Qwen3-32B (61 GB, bfloat16) — A100-80GB, 8×NVMe RAID-0, cold cache

| Method | Wall Time | Effective BW | Speedup |
|---|---|---|---|
| HF `from_pretrained` | 42.94s | 1.42 GB/s | 1.0x |
| Run:AI stream (CPU clone) | 26.49s | 2.30 GB/s | 1.6x |
| **Run:AI GPU-direct** | **19.63s** | **3.11 GB/s** | **2.2x** |

### Profiling Breakdown (Qwen3-32B)

| | HF | Stream (CPU) | GPU-direct |
|---|---|---|---|
| `cudaMemcpyAsync` CPU time | 147.93s | 51.76s | **8.17s** |
| Disk avg BW | 1.42 GB/s | 2.30 GB/s | **3.11 GB/s** |
| Disk utilization | 98% | 39% | **57%** |

## Running Benchmarks

```bash
# Quick comparison
CUDA_VISIBLE_DEVICES=0 python benchmark_loading.py \
  --model Qwen/Qwen3-8B --gpus 0 \
  --experiments hf runai_stream runai_gpu_direct --no-verify

# Full sweep (6 concurrency/worker configs × N repeats)
python benchmark_loading.py --model Qwen/Qwen3-32B --repeats 3

# Disk I/O timeline
CUDA_VISIBLE_DEVICES=0 python profile_io_timeline.py \
  --model Qwen/Qwen3-8B --experiment runai_gpu_direct --disk-device md0

# Full profiler (torch.profiler + iostat + cache stats)
CUDA_VISIBLE_DEVICES=0 python profile_loading.py \
  --model Qwen/Qwen3-8B --experiment runai_gpu_direct --disk-device md0
```

See [PROFILING.md](PROFILING.md) for detailed profiling methodology and how to interpret results.

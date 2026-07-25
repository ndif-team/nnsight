# Run:AI Model Loading with GPU-Direct Tensor Placement

Fast model loading for nnsight's `LanguageModel` via [run:ai model streamer](https://github.com/run-ai/runai-model-streamer) with direct buffer-to-GPU tensor placement, bypassing CPU intermediaries.

## Loading Paths

### 1. HF `from_pretrained` (baseline)

Standard HuggingFace loading: safetensors files are memory-mapped, tensors are materialized via 4 KB page faults, then HF worker threads call `.to(device)` for GPU placement. Bottlenecked by mmap read granularity and synchronous `cudaMemcpyAsync` blocking on each worker thread.

### 2. Run:AI GPU-Direct (default)

The loader thread resolves `device_map` before streaming begins, then copies tensors directly from the Run:AI buffer to the target GPU via `.to(device=cuda:N, dtype=dtype)`. When HF workers later call `.to()`, it's a no-op since the tensor is already on the correct device and dtype. The Run:AI streamer's background I/O threads read the next shard from disk concurrently with the blocking `.to()`, providing natural overlap between disk reads and GPU transfers.

```
HF baseline:       disk → mmap page faults → CPU → HF .to(cuda) → GPU
Run:AI GPU-direct: disk → run:ai buffer → .to(cuda) → GPU cache → HF .to() [no-op]
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

- **Quantization-aware device map:** When the model config contains `quantization_config`, the loader creates an `HfQuantizer` and runs `preprocess_model` on the meta model before computing the device map. This ensures `compute_module_sizes` sees the correct quantized parameter shapes (e.g., MXFP4 expert weights) instead of overestimating at full-precision sizes.

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

# Tune Run:AI I/O concurrency (default 16)
model = LanguageModel("Qwen/Qwen3-8B", device_map="auto", dispatch=True,
                      concurrency=32)
```

If `runai-model-streamer` is not installed, loading falls back to `from_pretrained` silently.

## Benchmark Results

### 8× A100-80GB PCIe, RAID /dev/md0, cold cache, best concurrency config

| Model | Type | Size | HF (s) | gpu_direct (s) | Speedup |
|---|---|---|---|---|---|
| Qwen/Qwen3-8B | Dense (2 GPU) | 15.3 GB | 10.31 | **5.04** | **2.0×** |
| Qwen/Qwen3-30B-A3B | MoE (8 GPU) | 56.9 GB | 49.26 | **23.76** | **2.1×** |
| openai/gpt-oss-120b | MXFP4 quantized (8 GPU) | 60.8 GB | 49.70 | **17.73** | **2.8×** |

## Running Benchmarks

```bash
# Quick comparison (hf vs gpu_direct)
python benchmark_loading.py \
  --model Qwen/Qwen3-8B --gpus 0 \
  --experiments hf gpu_direct --no-verify

# Full sweep with multiple concurrency/worker configs
python benchmark_loading.py --model Qwen/Qwen3-30B-A3B --gpus 0,1,2,3 --repeats 3
```

# Profiling Model Loading Performance

How to profile the nnsight model loading pipeline — disk I/O, CPU work, GPU transfers, and cache behavior. This guide is environment-agnostic; adapt device names and paths to your setup.

## What We Measure

There are four independent bottlenecks in model loading:

1. **Disk I/O** — how fast we read safetensors from storage
2. **CPU work** — cloning tensors, dtype casts, framework overhead
3. **GPU transfer** — `cudaMemcpyAsync` from CPU/buffer to GPU
4. **Thread coordination** — HF workers waiting for tensors, lock contention

Each tool below targets a specific bottleneck. Use them together to find which stage dominates wall time.

## Tool 1: Disk I/O Timeline (`/sys/class/block/<dev>/stat`)

**What it shows:** Per-interval read bandwidth and disk utilization percentage. Reveals whether the disk is busy or idle during loading.

**How it works:** A background thread polls `/sys/class/block/<dev>/stat` every 50-100ms, reading the cumulative `sectors_read` field. Delta between samples gives instantaneous bandwidth.

**Finding your block device:**

```bash
# List block devices
lsblk -d -o NAME,SIZE,TYPE,ROTA

# If using RAID, check /proc/mdstat
cat /proc/mdstat

# Find which device holds the HF cache
df -h ~/.cache/huggingface/
```

Common setups:
- Single NVMe: `nvme0n1`
- RAID-0 array: `md0` (aggregates all member drives)
- Network FS (Lustre/NFS): no local block device — use `iostat` or network stats instead

**Usage:**

```bash
# profile_io_timeline.py samples disk stats and prints a visual timeline
CUDA_VISIBLE_DEVICES=0 python profile_io_timeline.py \
  --model Qwen/Qwen3-8B --experiment runai_gpu_direct \
  --disk-device md0 --sample-ms 100
```

**Reading the output:**

```
  Time   Read MB   BW GB/s  Bar
   1.6s       924      7.30  █████████████████████████████████████
   1.7s      1044     10.17  ████████████████████████████████████████
   ...
   2.4s         0      0.00  ·       ← disk idle, GPU transfer in progress
   2.5s         0      0.00  ·
   2.7s       237      2.31  ███████████
```

Idle gaps between bursts mean something downstream (GPU copy, CPU clone) is blocking the loader from issuing the next read. The goal is to shrink these gaps.

**Key metrics:**
- `active %` — fraction of wall time the disk is reading (higher = better utilization)
- `peak BW` — maximum instantaneous bandwidth (shows hardware capability)
- `avg BW` — total_read / wall_time (effective throughput)

## Tool 2: Cache Stats (Monkey-Patched `build_lazy_state_dict`)

**What it shows:** Time breakdown inside the Run:AI streaming cache — how much time is spent waiting for disk, cloning to CPU, copying to GPU, and waiting for consumers.

**How it works:** Monkey-patch `build_lazy_state_dict` to capture the `RunAIShardCache` object after construction. After loading completes, read its `stats_*` fields.

```python
import nnsight.modeling.loader as loader_mod

captured = {}
orig = loader_mod.build_lazy_state_dict

def patched(*a, **kw):
    result = orig(*a, **kw)
    for v in result.values():
        captured["cache"] = v._cache
        break
    return result

loader_mod.build_lazy_state_dict = patched

# ... load model ...

loader_mod.build_lazy_state_dict = orig  # restore

cache = captured["cache"]
print(f"shard wall:    {cache.stats_shard_wall_s:.2f}s")
print(f"io wait:       {cache.stats_io_wait_s:.2f}s")
print(f"clone:         {cache.stats_clone_s:.2f}s")
print(f"gpu copy:      {cache.stats_gpu_copy_s:.2f}s")
print(f"consumer wait: {cache.stats_consumer_wait_s:.2f}s")
print(f"tensors:       {cache.stats_pop_count}")
```

**Interpreting results:**

| Stat | Meaning |
|---|---|
| `shard_wall_s` | Total wall time of all `_stream_shard_incremental` calls |
| `io_wait_s` | Time between `streamer.get_tensors()` yields — uncovered disk I/O |
| `clone_s` | Time spent in `.clone()` (CPU-target tensors) |
| `gpu_copy_s` | Time spent in `.to(device=cuda)` (GPU-direct path) |
| `consumer_wait_s` | Cumulative time HF workers spent waiting on the Condition variable for their tensor to appear |

**What to look for:**
- `io_wait >> gpu_copy` → disk-bound. Increase `concurrency` or use faster storage.
- `gpu_copy >> io_wait` → GPU-transfer-bound (PCIe bandwidth). Disk is underutilized.
- `consumer_wait` is high → workers spend most time waiting, meaning the loader is the bottleneck (this is normal — it means the pipeline is working correctly with the loader as the single producer).

## Tool 3: `torch.profiler` (CPU ops + CUDA events)

**What it shows:** Which PyTorch operations consume the most CPU time. The key operation to look for is `cudaMemcpyAsync` — this is the synchronous CPU time spent initiating GPU transfers.

**How it works:**

```python
activities = [
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]
with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
    model = LanguageModel(...)

# Summary
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

# Or export for visual inspection
prof.export_chrome_trace("trace.json")
# Open at https://ui.perfetto.dev
```

**Key operations to watch:**

| Operation | What it means |
|---|---|
| `cudaMemcpyAsync` | CPU→GPU transfer (this is the big one) |
| `aten::clone` | Tensor cloning (CPU→CPU) |
| `aten::copy_` | In-place copy (used by HF's weight assignment) |
| `aten::_to_copy` | `.to(device, dtype)` implementation |

**Interpreting `cudaMemcpyAsync` CPU time:**

This is the time the CPU thread spends **blocked** waiting for the GPU DMA transfer to complete. Despite the name "async", it blocks the calling thread when there's no free DMA channel or when the source memory isn't pinned. In HF loading, each worker thread calls `.to(device)` which triggers this.

- **HF baseline:** ~148s cumulative across 4 workers (each transfer blocks the thread)
- **Run:AI CPU clone:** ~52s (same transfers, but tensors arrive faster from cache)
- **Run:AI GPU-direct:** ~8s (most transfers already done by the loader; HF's `.to()` is a no-op)

## Tool 4: `iostat` (Per-Second Disk Stats)

**What it shows:** Read throughput and utilization percentage per second. Less granular than Tool 1 but easier to set up.

**Usage:**

```bash
# In a separate terminal, start before loading
iostat -x -d md0 1

# Key columns:
#   rkB/s  — read bandwidth in KB/s
#   %util  — percentage of time the device was busy
```

`profile_loading.py` runs this automatically via `IOStatSampler`.

## Page Cache Control

**Critical for reproducible benchmarks.** The kernel page cache can make subsequent loads appear instant.

**Evicting model pages (no sudo needed):**

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = snapshot_download(model_id, local_files_only=True)
for path in Path(model_dir).glob("*.safetensors"):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
```

This calls `posix_fadvise(FADV_DONTNEED)` per file, which tells the kernel to drop those pages from cache. Works per-inode, no sudo required.

**Nuclear option (requires sudo):**

```bash
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
```

**Warm cache benchmarks:** Read all shards into page cache before timing:

```python
for path in shard_paths:
    with open(path, "rb") as f:
        while f.read(16 * 1024 * 1024):
            pass
```

## Controlling HF Worker Threads

HF transformers v5 uses `GLOBAL_WORKERS` threads for parallel weight materialization. Patch it before loading:

```python
import transformers.core_model_loading as cml
cml.GLOBAL_WORKERS = 4  # default is 4
```

This controls how many threads call `_materialize_copy` in parallel. With GPU-direct, these workers mostly just wait since tensors are already on GPU.

## Putting It All Together

`profile_loading.py` combines all four tools:

```bash
# Run:AI GPU-direct profiling
CUDA_VISIBLE_DEVICES=0 python profile_loading.py \
  --model Qwen/Qwen3-8B --experiment runai_gpu_direct \
  --disk-device md0 --workers 4 --concurrency 16

# Run:AI CPU-clone profiling
CUDA_VISIBLE_DEVICES=0 python profile_loading.py \
  --model Qwen/Qwen3-8B --experiment runai_stream \
  --disk-device md0

# HF baseline
CUDA_VISIBLE_DEVICES=0 python profile_loading.py \
  --model Qwen/Qwen3-8B --experiment hf \
  --disk-device md0

# Export chrome trace for visual analysis
CUDA_VISIBLE_DEVICES=0 python profile_loading.py \
  --model Qwen/Qwen3-8B --experiment runai_gpu_direct \
  --trace trace_gpu_direct.json
```

## Reference: Profiling Results (Qwen3-32B, A100-80GB, 8×NVMe RAID-0)

These numbers are from a specific environment (nagoya server, March 2026) and serve as a baseline for comparison. Your numbers will differ.

| | HF | Run:AI stream | GPU-direct |
|---|---|---|---|
| Wall time | 42.89s | 27.21s | **17.52s** |
| Effective BW | 1.53 GB/s | 2.41 GB/s | **3.74 GB/s** |
| `cudaMemcpyAsync` CPU | 147.93s | 51.76s | **8.17s** |
| Disk active | 98% | 39% | **63%** |
| Disk peak BW | 2.0 GB/s | 10.7 GB/s | 10.3 GB/s |
| Disk avg BW | 1.42 GB/s | 2.24 GB/s | **3.48 GB/s** |

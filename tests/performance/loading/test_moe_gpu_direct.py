"""Benchmark peak GPU memory for MoE loading: hf vs gpu_direct vs gpu_direct_lazy.

Qwen3-30B-A3B (128 experts, 56.87 GiB bf16) on a single GPU.
Tracks peak GPU memory via torch.cuda.max_memory_allocated() to show
how much transient overhead each loading path requires beyond the final
model size.

Expected:
  hf              — low overhead  (safetensors mmap reads one param at a time)
  gpu_direct      — high overhead (eager cache preloads entire shards onto GPU)
  gpu_direct_lazy — low overhead  (streams one tensor at a time)
"""

import gc
import os
import time
from pathlib import Path

import torch


MODEL_ID = "Qwen/Qwen3-30B-A3B"
GPU_ID = 7
MODEL_SIZE_GIB = 56.87


def evict_model_pages(model_id, revision="main"):
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(model_id, revision=revision, local_files_only=True)
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def run_experiment(name, extra_kwargs):
    from nnsight import LanguageModel

    evict_model_pages(MODEL_ID)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(GPU_ID)
    torch.cuda.reset_peak_memory_stats(GPU_ID)

    # No balanced budget — just restrict to this GPU
    max_memory = {i: 0 for i in range(torch.cuda.device_count())}
    max_memory[GPU_ID] = int(torch.cuda.get_device_properties(GPU_ID).total_memory * 0.9)

    try:
        t0 = time.perf_counter()
        model = LanguageModel(
            MODEL_ID,
            device_map="auto",
            max_memory=max_memory,
            dispatch=True,
            **extra_kwargs,
        )
        torch.cuda.synchronize(GPU_ID)
        wall = time.perf_counter() - t0

        final_alloc = torch.cuda.memory_allocated(GPU_ID) / 1024**3
        peak_alloc = torch.cuda.max_memory_allocated(GPU_ID) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(GPU_ID) / 1024**3
        overhead = peak_alloc - final_alloc

        print(f"  [{name}] OK — {wall:.1f}s")
        print(f"    final_alloc:  {final_alloc:.2f} GiB")
        print(f"    peak_alloc:   {peak_alloc:.2f} GiB")
        print(f"    peak_reserved:{peak_reserved:.2f} GiB")
        print(f"    overhead:     {overhead:.2f} GiB  (peak - final)")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {"wall": wall, "final": final_alloc, "peak": peak_alloc,
                "reserved": peak_reserved, "overhead": overhead}

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        peak_alloc = torch.cuda.max_memory_allocated(GPU_ID) / 1024**3
        print(f"  [{name}] FAILED — {type(e).__name__}")
        print(f"    peak_alloc at failure: {peak_alloc:.2f} GiB")
        gc.collect()
        torch.cuda.empty_cache()
        return {"wall": 0, "final": 0, "peak": peak_alloc, "reserved": 0,
                "overhead": 0, "error": str(e)[:80]}


def main():
    print(f"Model:  {MODEL_ID} ({MODEL_SIZE_GIB:.1f} GiB)")
    print(f"GPU:    {GPU_ID} ({torch.cuda.get_device_name(GPU_ID)})")
    free = torch.cuda.mem_get_info(GPU_ID)[0] / 1024**3
    print(f"Free:   {free:.1f} GiB")

    # Init CUDA context on the target GPU
    torch.zeros(1, device=f"cuda:{GPU_ID}")

    experiments = [
        ("hf",              {"load_format": "from_pretrained"}),
        ("gpu_direct",      {"concurrency": 8}),
        ("gpu_direct_lazy", {"concurrency": 8, "lazy": True}),
    ]

    results = {}
    for name, kwargs in experiments:
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        results[name] = run_experiment(name, kwargs)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Experiment':<20} {'Wall':>6} {'Final':>8} {'Peak':>8} {'Overhead':>10} "
          f"{'Reserved':>10} {'Status':>8}")
    print("-" * 80)
    for name, r in results.items():
        status = "OK" if "error" not in r else "OOM"
        print(f"{name:<20} {r['wall']:>5.1f}s {r['final']:>7.2f}G {r['peak']:>7.2f}G "
              f"{r['overhead']:>9.2f}G {r['reserved']:>9.2f}G {status:>8}")
    print(f"\n  Model size:  {MODEL_SIZE_GIB:.2f} GiB")
    print(f"  Overhead   = peak_alloc - final_alloc (transient live tensors)")
    print(f"  Reserved   = peak CUDA allocator reservation (includes fragmentation)")


if __name__ == "__main__":
    main()

"""Profile model loading: combined I/O, CPU, and GPU timeline.

Combines:
  - iostat: per-second disk utilization (is disk busy or idle?)
  - torch.profiler: CPU ops + CUDA memcpy timeline
  - cache stats: loader vs consumer wait breakdown

Usage:
  CUDA_VISIBLE_DEVICES=0 python profile_loading.py --model Qwen/Qwen3-8B --experiment runai_stream
  CUDA_VISIBLE_DEVICES=0 python profile_loading.py --model Qwen/Qwen3-8B --experiment hf
"""

import argparse
import gc
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

import torch


def evict_model_pages(model_id, revision="main"):
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(model_id, revision=revision, local_files_only=True)
    shard_paths = sorted(Path(model_dir).glob("*.safetensors"))
    total = 0
    for path in shard_paths:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
            total += size
        finally:
            os.close(fd)
    print(f"  Evicted {len(shard_paths)} shards ({total / 1024**3:.1f} GB)")


def build_max_memory(gpu_ids):
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        if i in gpu_ids:
            mem = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(mem * 0.9)
        else:
            max_memory[i] = 0
    return max_memory


def patch_hf_workers(n):
    try:
        import transformers.core_model_loading as cml

        original = cml.GLOBAL_WORKERS
        cml.GLOBAL_WORKERS = n
        return lambda: setattr(cml, "GLOBAL_WORKERS", original)
    except (ImportError, AttributeError):
        pass
    return lambda: None


class IOStatSampler:
    """Sample iostat in a background thread at 1-second intervals."""

    def __init__(self, device="md0"):
        self.device = device
        self.samples = []  # (timestamp, rMB_s, wMB_s, util_pct)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        try:
            proc = subprocess.Popen(
                ["iostat", "-x", "-d", self.device, "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.kill()
                    break
                parts = line.split()
                if not parts or parts[0] != self.device:
                    continue
                try:
                    # iostat -x columns: Device r/s rkB/s ... %util
                    rMB_s = float(parts[2]) / 1024  # rkB/s -> MB/s
                    wMB_s = float(parts[8]) / 1024 if len(parts) > 8 else 0
                    util = float(parts[-1])
                    self.samples.append(
                        (time.perf_counter(), rMB_s, wMB_s, util)
                    )
                except (ValueError, IndexError):
                    continue
            proc.wait()
        except FileNotFoundError:
            pass  # iostat not installed

    def print_summary(self, t0, t_end):
        if not self.samples:
            print("\n  [iostat] No samples (iostat not available?)")
            return
        print(f"\n  [iostat] Disk utilization timeline ({self.device}):")
        print(f"  {'Time':>6s}  {'Read MB/s':>10s}  {'%util':>6s}")
        print(f"  {'------':>6s}  {'----------':>10s}  {'------':>6s}")
        for ts, rMB, wMB, util in self.samples:
            elapsed = ts - t0
            if elapsed < -1 or elapsed > (t_end - t0) + 2:
                continue
            print(f"  {elapsed:6.1f}s  {rMB:10.1f}  {util:6.1f}")


def profile_load(model_id, gpu_ids, experiment, concurrency, workers,
                 revision, disk_device, trace_file):
    from nnsight import LanguageModel

    max_memory = build_max_memory(gpu_ids)
    restore = patch_hf_workers(workers)

    # Build kwargs per experiment
    extra = {}
    if experiment == "hf":
        extra = {"load_format": "from_pretrained"}
    elif experiment == "runai_stream":
        extra = {"concurrency": concurrency, "gpu_direct": False}
    elif experiment == "runai_gpu_direct":
        extra = {"concurrency": concurrency}

    # Start iostat
    iostat = IOStatSampler(device=disk_device)
    iostat.start()

    # Evict cache
    evict_model_pages(model_id, revision)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"\n  Loading {experiment} (conc={concurrency}, workers={workers})...")

    # Profile with torch.profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    t0 = time.perf_counter()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        model = LanguageModel(
            model_id,
            device_map="auto",
            max_memory=max_memory,
            revision=revision,
            dispatch=True,
            **extra,
        )

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    iostat.stop()
    restore()

    # Print results
    model_size_gb = sum(
        p.numel() * p.element_size() for p in model._model.parameters()
    ) / 1024**3

    print(f"\n{'=' * 60}")
    print(f"  Experiment:  {experiment}")
    print(f"  Wall time:   {wall:.2f}s")
    print(f"  Bandwidth:   {model_size_gb / wall:.2f} GB/s")
    print(f"  Model:       {model_size_gb:.1f} GB on GPU")

    # torch.profiler summary
    print(f"\n  [torch.profiler] Top CPU operations:")
    table = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=20
    )
    print(table)

    # CUDA summary
    cuda_events = prof.key_averages()
    memcpy_time = sum(
        e.cuda_time
        for e in cuda_events
        if "memcpy" in e.key.lower() or "Memcpy" in e.key
    )
    total_cuda = sum(e.cuda_time for e in cuda_events)
    print(f"\n  [CUDA] Total CUDA time: {total_cuda / 1e6:.2f}s")
    print(f"  [CUDA] Memcpy time:     {memcpy_time / 1e6:.2f}s")

    # iostat timeline
    iostat.print_summary(t0, t0 + wall)

    # Export chrome trace
    if trace_file:
        prof.export_chrome_trace(trace_file)
        print(f"\n  Chrome trace saved to: {trace_file}")
        print(f"  Open with: chrome://tracing or https://ui.perfetto.dev")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


def profile_load_with_cache_stats(model_id, gpu_ids, experiment, concurrency,
                                  workers, revision, disk_device, trace_file):
    """Load with monkey-patched build_lazy_state_dict to capture cache stats."""
    import nnsight.modeling.loader as loader_mod

    max_memory = build_max_memory(gpu_ids)
    restore = patch_hf_workers(workers)

    captured_cache = {}

    # Monkey-patch to capture the cache object
    orig_build = loader_mod.build_lazy_state_dict

    def patched_build(*args, **kwargs):
        result = orig_build(*args, **kwargs)
        for v in result.values():
            captured_cache["cache"] = v._cache
            break
        return result

    loader_mod.build_lazy_state_dict = patched_build

    if experiment == "runai_stream":
        extra = {"concurrency": concurrency, "gpu_direct": False}
    else:  # runai_gpu_direct
        extra = {"concurrency": concurrency}

    iostat = IOStatSampler(device=disk_device)
    iostat.start()

    evict_model_pages(model_id, revision)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"\n  Loading {experiment} (conc={concurrency}, workers={workers})...")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    t0 = time.perf_counter()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        from nnsight import LanguageModel

        model = LanguageModel(
            model_id,
            device_map="auto",
            max_memory=max_memory,
            revision=revision,
            dispatch=True,
            **extra,
        )

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    iostat.stop()

    loader_mod.build_lazy_state_dict = orig_build
    restore()

    model_size_gb = sum(
        p.numel() * p.element_size() for p in model._model.parameters()
    ) / 1024**3

    print(f"\n{'=' * 60}")
    print(f"  Experiment:  {experiment}")
    print(f"  Wall time:   {wall:.2f}s")
    print(f"  Bandwidth:   {model_size_gb / wall:.2f} GB/s")

    # Cache stats
    cache = captured_cache.get("cache")
    if cache:
        print(f"\n  [cache stats]")
        print(f"    tensors served:    {cache.stats_pop_count}")
        print(f"    shard wall time:   {cache.stats_shard_wall_s:.2f}s  (total time in _stream_shard_incremental)")
        print(f"      io wait:         {cache.stats_io_wait_s:.2f}s  (uncovered — streamer yield blocked on disk)")
        print(f"      clone:           {cache.stats_clone_s:.2f}s")
        print(f"      gpu copy:        {cache.stats_gpu_copy_s:.2f}s")
        hidden_io = cache.stats_shard_wall_s - cache.stats_io_wait_s - cache.stats_clone_s - cache.stats_gpu_copy_s
        print(f"      notify+lock:     {hidden_io:.2f}s  (lock acquire + notify_all + overhead)")
        print(f"    consumer wait:     {cache.stats_consumer_wait_s:.2f}s  (cumulative across {workers} workers)")

    # torch.profiler
    print(f"\n  [torch.profiler] Top CPU operations:")
    table = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=15
    )
    print(table)

    cuda_events = prof.key_averages()
    memcpy_time = sum(
        e.cuda_time
        for e in cuda_events
        if "memcpy" in e.key.lower() or "Memcpy" in e.key
    )
    total_cuda = sum(e.cuda_time for e in cuda_events)
    print(f"\n  [CUDA] Total CUDA time: {total_cuda / 1e6:.2f}s")
    print(f"  [CUDA] Memcpy time:     {memcpy_time / 1e6:.2f}s")

    iostat.print_summary(t0, t0 + wall)

    if trace_file:
        prof.export_chrome_trace(trace_file)
        print(f"\n  Chrome trace: {trace_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Profile model loading pipeline")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--experiment",
        default="runai_stream",
        choices=["hf", "runai_stream", "runai_gpu_direct"],
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--disk-device", default="md0")
    parser.add_argument("--trace", default=None,
                        help="Output chrome trace JSON file")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]

    print(f"Model:       {args.model}")
    print(f"Experiment:  {args.experiment}")
    print(f"GPUs:        {gpu_ids}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Workers:     {args.workers}")
    print(f"Disk:        {args.disk_device}")

    if args.experiment == "hf":
        profile_load(
            args.model, gpu_ids, args.experiment, args.concurrency,
            args.workers, args.revision, args.disk_device, args.trace,
        )
    else:  # runai_stream or runai_gpu_direct
        profile_load_with_cache_stats(
            args.model, gpu_ids, args.experiment, args.concurrency,
            args.workers, args.revision, args.disk_device, args.trace,
        )


if __name__ == "__main__":
    main()

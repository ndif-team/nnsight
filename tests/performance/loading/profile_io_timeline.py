"""Profile disk I/O timeline during model loading.

Samples /sys/class/block/<device>/stat at high frequency to show
when the disk is active vs idle during the loading pipeline.

Usage:
  CUDA_VISIBLE_DEVICES=0 python profile_io_timeline.py --model Qwen/Qwen3-8B --experiment runai_stream
  CUDA_VISIBLE_DEVICES=0 python profile_io_timeline.py --model Qwen/Qwen3-8B --experiment hf
"""

import argparse
import gc
import os
import time
import threading
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Disk stat sampler
# ---------------------------------------------------------------------------

class DiskStatSampler:
    """Sample /sys/class/block/<dev>/stat at high frequency."""

    def __init__(self, device="md0", interval_ms=50):
        self._path = f"/sys/class/block/{device}/stat"
        self._interval = interval_ms / 1000.0
        self._samples = []  # (timestamp, sectors_read, read_ms)
        self._stop = threading.Event()
        self._thread = None

    def _read_stat(self):
        with open(self._path) as f:
            fields = f.read().split()
        return int(fields[2]), int(fields[3])  # sectors_read, read_ticks_ms

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            try:
                sectors, read_ms = self._read_stat()
                self._samples.append((time.perf_counter(), sectors, read_ms))
            except Exception:
                pass
            self._stop.wait(self._interval)

    def get_timeline(self, t0):
        """Return list of (elapsed_s, delta_MB, delta_read_ms, bandwidth_MB_s)."""
        timeline = []
        for i in range(1, len(self._samples)):
            ts, sec, rms = self._samples[i]
            ts_prev, sec_prev, rms_prev = self._samples[i - 1]
            dt = ts - ts_prev
            if dt <= 0:
                continue
            delta_bytes = (sec - sec_prev) * 512
            delta_mb = delta_bytes / (1024 * 1024)
            bw = delta_mb / dt
            elapsed = ts - t0
            timeline.append((elapsed, delta_mb, rms - rms_prev, bw))
        return timeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evict_model_pages(model_id, revision="main"):
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(model_id, revision=revision, local_files_only=True)
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def build_max_memory(gpu_ids):
    mm = {}
    for i in range(torch.cuda.device_count()):
        if i in gpu_ids:
            mm[i] = int(torch.cuda.get_device_properties(i).total_memory * 0.9)
        else:
            mm[i] = 0
    return mm


def patch_hf_workers(n):
    try:
        import transformers.core_model_loading as cml
        original = cml.GLOBAL_WORKERS
        cml.GLOBAL_WORKERS = n
        return lambda: setattr(cml, "GLOBAL_WORKERS", original)
    except (ImportError, AttributeError):
        return lambda: None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--experiment", default="runai_gpu_direct",
                        choices=["hf", "runai_stream", "runai_gpu_direct"])
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--disk-device", default="md0")
    parser.add_argument("--sample-ms", type=int, default=50)
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    max_memory = build_max_memory(gpu_ids)
    restore = patch_hf_workers(args.workers)

    extra = {}
    if args.experiment == "hf":
        extra = {"load_format": "from_pretrained"}
    elif args.experiment == "runai_stream":
        extra = {"concurrency": args.concurrency, "gpu_direct": False}
    elif args.experiment == "runai_gpu_direct":
        extra = {"concurrency": args.concurrency}

    # Monkey-patch to capture cache
    import nnsight.modeling.loader as loader_mod
    captured = {}
    orig_build = loader_mod.build_lazy_state_dict
    def patched_build(*a, **kw):
        result = orig_build(*a, **kw)
        for v in result.values():
            captured["cache"] = v._cache
            break
        return result
    loader_mod.build_lazy_state_dict = patched_build

    sampler = DiskStatSampler(device=args.disk_device, interval_ms=args.sample_ms)

    evict_model_pages(args.model, args.revision)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"Model:       {args.model}")
    print(f"Experiment:  {args.experiment}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Workers:     {args.workers}")
    print(f"Disk:        {args.disk_device} (sampling every {args.sample_ms}ms)")
    print()

    sampler.start()
    t0 = time.perf_counter()

    from nnsight import LanguageModel
    model = LanguageModel(
        args.model, device_map="auto", max_memory=max_memory,
        revision=args.revision, dispatch=True, **extra,
    )

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sampler.stop()

    loader_mod.build_lazy_state_dict = orig_build
    restore()

    # Print timeline
    timeline = sampler.get_timeline(t0)
    print(f"Wall time: {wall:.2f}s")
    print()

    # Cache stats
    cache = captured.get("cache")
    if cache:
        print(f"[cache] shard_wall={cache.stats_shard_wall_s:.2f}s  "
              f"io_wait={cache.stats_io_wait_s:.2f}s  "
              f"clone={cache.stats_clone_s:.2f}s  "
              f"gpu_copy={cache.stats_gpu_copy_s:.2f}s")
        print()

    # Disk I/O timeline
    IDLE_THRESHOLD_MB = 10  # below this MB/interval = idle
    print(f"{'Time':>6s}  {'Read MB':>8s}  {'BW MB/s':>9s}  {'Status'}")
    print(f"{'------':>6s}  {'--------':>8s}  {'---------':>9s}  {'------'}")

    total_read_mb = 0
    io_active_s = 0
    io_idle_s = 0
    prev_elapsed = 0

    for elapsed, delta_mb, delta_rms, bw in timeline:
        if elapsed < -0.5 or elapsed > wall + 0.5:
            continue
        total_read_mb += delta_mb
        dt = elapsed - prev_elapsed if prev_elapsed else args.sample_ms / 1000
        prev_elapsed = elapsed

        if delta_mb > IDLE_THRESHOLD_MB:
            status = "ACTIVE" + " " + "█" * min(50, int(bw / 100))
            io_active_s += dt
        else:
            status = "idle"
            io_idle_s += dt

        print(f"{elapsed:6.2f}s  {delta_mb:8.1f}  {bw:9.1f}  {status}")

    print()
    print(f"Total read:  {total_read_mb / 1024:.2f} GB")
    print(f"I/O active:  {io_active_s:.2f}s")
    print(f"I/O idle:    {io_idle_s:.2f}s")
    print(f"Disk util:   {io_active_s / (io_active_s + io_idle_s) * 100:.0f}%" if (io_active_s + io_idle_s) > 0 else "")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

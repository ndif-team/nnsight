"""Benchmark: nnsight LanguageModel loading — two paths compared.

Compares wall-clock time, peak GPU/CPU memory, and output correctness for
loading HuggingFace models through nnsight's LanguageModel interface:

  hf          — standard from_pretrained (safetensors mmap → threaded loading)
  runai_lazy  — run:ai streaming (read + pthreads), incremental tensor-by-tensor loading

Page cache invalidation between experiments:
  By default, uses posix_fadvise(FADV_DONTNEED) on model shard files to evict
  their pages from the kernel page cache. This is reliable and requires no sudo.
  With --junk-drop-caches, reads a large junk file to pressure the page cache
  (unreliable due to Linux active/inactive list behavior).
  With --sudo-drop-caches, uses 'echo 3 > /proc/sys/vm/drop_caches' instead.
  With --no-drop-caches, skips invalidation entirely (warm-cache runs).

Usage:
  # Cold cache (default): fadvise evicts model pages before each run
  python benchmark_loading.py --model meta-llama/Llama-3.1-8B --gpus 0,1

  # Warm cache: warmup reads shards into page cache, then no eviction
  python benchmark_loading.py --model meta-llama/Llama-3.1-8B --warmup --no-drop-caches

  # Specific experiments
  python benchmark_loading.py --model Qwen/Qwen2.5-7B-Instruct --experiments hf runai_lazy

  # Both methods, 3 repeats, JSON output
  python benchmark_loading.py --model meta-llama/Llama-3.1-8B --repeats 3 --output results.json
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    experiment: str
    config: dict
    wall_time_s: float
    peak_gpu_mem_mb: float = 0.0
    peak_rss_mb: float = 0.0
    peak_private_mb: float = 0.0
    disk_read_gib: Optional[float] = None
    net_rx_gib: Optional[float] = None
    net_tx_gib: Optional[float] = None
    error: Optional[str] = None


def _read_diskstats(device: str = "nvme0n1") -> Optional[float]:
    """Read cumulative bytes read for *device* from sysfs.  Returns GiB."""
    stat_path = f"/sys/class/block/{device}/stat"
    try:
        with open(stat_path) as f:
            fields = f.read().split()
        sectors_read = int(fields[2])  # 3rd field
        return sectors_read * 512 / 1024**3
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _read_iface_bytes(iface: str = "hsn0") -> Optional[tuple[float, float]]:
    """Read (rx_gib, tx_gib) for *iface* from /proc/net/dev."""
    try:
        with open("/proc/net/dev") as f:
            for line in f.readlines()[2:]:
                name, data = line.split(":", 1)
                if name.strip() == iface:
                    fields = data.split()
                    rx = int(fields[0]) / 1024**3
                    tx = int(fields[8]) / 1024**3
                    return rx, tx
    except (FileNotFoundError, IndexError, ValueError):
        pass
    return None


def get_process_rss_mb() -> float:
    """Current process RSS in MB via /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def get_private_dirty_mb() -> float:
    """Private dirty memory in MB via /proc/self/smaps_rollup.

    Returns only Private_Dirty — pages the process allocated and wrote to.
    Excludes Private_Clean (which includes MAP_PRIVATE mmap read-only pages
    like safetensors mmap that don't actually consume extra physical RAM
    beyond the page cache).
    """
    private_kb = 0
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if line.startswith("Private_Dirty:"):
                    private_kb += int(line.split()[1])
    except Exception:
        return get_process_rss_mb()  # fallback
    return private_kb / 1024


class PeakMemMonitor:
    """Context manager that polls process memory in a background thread.

    Tracks both RSS (total resident, including shared mmap pages) and
    private memory (excluding shared pages like safetensors mmap).
    Records the peak of each observed during the context.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_rss_mb: float = 0.0
        self.peak_private_mb: float = 0.0
        self._stop = False
        self._thread = None

    def _poll(self):
        while not self._stop:
            rss = get_process_rss_mb()
            private = get_private_dirty_mb()
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
            if private > self.peak_private_mb:
                self.peak_private_mb = private
            time.sleep(self.interval)

    def __enter__(self):
        import threading
        self.peak_rss_mb = get_process_rss_mb()
        self.peak_private_mb = get_private_dirty_mb()
        self._stop = False
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._thread.join(timeout=2)
        # One final sample
        rss = get_process_rss_mb()
        private = get_private_dirty_mb()
        if rss > self.peak_rss_mb:
            self.peak_rss_mb = rss
        if private > self.peak_private_mb:
            self.peak_private_mb = private


def get_gpu_mem_allocated_mb(gpu_ids: list[int]) -> float:
    return sum(torch.cuda.memory_allocated(i) / (1024 ** 2) for i in gpu_ids)


def build_max_memory(gpu_ids: list[int]) -> dict:
    """Build a max_memory dict restricting placement to the given GPUs."""
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        if i in gpu_ids:
            mem = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(mem * 0.9)
        else:
            max_memory[i] = 0
    return max_memory


def unload_model(model):
    """Fully unload a model and free GPU memory."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def resolve_model_size_gb(model_id: str, revision: str = "main") -> float:
    """Total size of model shard files in GB."""
    from huggingface_hub import snapshot_download

    try:
        model_dir = snapshot_download(model_id, revision=revision, local_files_only=True)
    except Exception:
        return 0.0
    paths = list(Path(model_dir).glob("*.safetensors"))
    if not paths:
        paths = list(Path(model_dir).glob("*.bin"))
    return sum(p.stat().st_size for p in paths) / (1024 ** 3)


# ---------------------------------------------------------------------------
# Page cache invalidation
# ---------------------------------------------------------------------------

JUNK_FILE_PATH = "/tmp/bench_loading_junk.bin"
JUNK_FILE_SIZE_GB = 512


def create_junk_file(path: str, size_gb: int):
    if os.path.exists(path):
        existing_gb = os.path.getsize(path) / (1024 ** 3)
        if existing_gb >= size_gb * 0.99:
            print(f"  [junk] Reusing existing {existing_gb:.0f} GB file at {path}")
            return
    print(f"  [junk] Creating {size_gb} GB junk file at {path}...")
    t0 = time.perf_counter()
    subprocess.run(
        ["dd", "if=/dev/zero", f"of={path}", "bs=1M", f"count={size_gb * 1024}",
         "status=progress"],
        check=True,
    )
    print(f"  [junk] Created in {time.perf_counter() - t0:.0f}s")


def invalidate_via_junk_file(path: str, num_threads: int = 16):
    if not os.path.exists(path):
        print(f"  [cache] Junk file not found: {path}")
        return
    file_size = os.path.getsize(path)
    size_gb = file_size / (1024 ** 3)
    chunk_size = file_size // num_threads
    print(f"  [cache] Reading {size_gb:.0f} GB junk file ({num_threads} threads)...")

    def _read_chunk(idx):
        offset = idx * chunk_size
        end = file_size if idx == num_threads - 1 else offset + chunk_size
        with open(path, "rb") as f:
            f.seek(offset)
            remaining = end - offset
            while remaining > 0:
                data = f.read(min(16 * 1024 * 1024, remaining))
                if not data:
                    break
                remaining -= len(data)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        list(pool.map(_read_chunk, range(num_threads)))
    elapsed = time.perf_counter() - t0
    print(f"  [cache] Done in {elapsed:.1f}s ({size_gb / elapsed:.1f} GB/s)")


def evict_model_pages(model_id: str, revision: str = "main"):
    """Drop page cache for model shard files using posix_fadvise(FADV_DONTNEED).

    This evicts pages by inode regardless of which process/function loaded them.
    Works without sudo and targets only the model files (not the entire cache).
    """
    from huggingface_hub import snapshot_download

    try:
        model_dir = snapshot_download(model_id, revision=revision, local_files_only=True)
    except Exception as e:
        print(f"  [cache] Could not locate model files: {e}")
        return

    shard_paths = sorted(Path(model_dir).glob("*.safetensors"))
    if not shard_paths:
        shard_paths = sorted(Path(model_dir).glob("*.bin"))
    if not shard_paths:
        print("  [cache] No shard files found to evict")
        return

    total_bytes = 0
    for path in shard_paths:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
            total_bytes += size
        finally:
            os.close(fd)
    print(f"  [cache] Evicted {len(shard_paths)} shard files "
          f"({total_bytes / 1024**3:.1f} GB) via FADV_DONTNEED")


def warm_model_pages(model_id: str, revision: str = "main",
                     num_threads: int = 16):
    """Read all model shard files into page cache (warmup for warm-cache runs)."""
    from huggingface_hub import snapshot_download

    try:
        model_dir = snapshot_download(model_id, revision=revision,
                                      local_files_only=True)
    except Exception as e:
        print(f"  [warmup] Could not locate model files: {e}")
        return

    shard_paths = sorted(Path(model_dir).glob("*.safetensors"))
    if not shard_paths:
        shard_paths = sorted(Path(model_dir).glob("*.bin"))
    if not shard_paths:
        print("  [warmup] No shard files found")
        return

    total_bytes = sum(p.stat().st_size for p in shard_paths)

    def _read_file(path):
        with open(path, "rb") as f:
            while f.read(16 * 1024 * 1024):
                pass

    t0 = time.perf_counter()
    print(f"  [warmup] Reading {len(shard_paths)} shard files "
          f"({total_bytes / 1024**3:.1f} GB) into page cache...")
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        list(pool.map(_read_file, shard_paths))
    elapsed = time.perf_counter() - t0
    bw = total_bytes / 1024**3 / elapsed if elapsed > 0 else 0
    print(f"  [warmup] Done in {elapsed:.1f}s ({bw:.1f} GB/s)")


def sudo_drop_caches():
    print("  [cache] Dropping page caches (sudo)...")
    subprocess.run(
        ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
        check=True, timeout=30,
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def _patch_hf_workers(n_workers: int):
    """Set the number of worker threads for from_pretrained shard loading.

    Supports both transformers v5 (GLOBAL_WORKERS) and v4 (env var).
    Returns a restore function.
    """
    try:
        import transformers.core_model_loading as cml
        original = cml.GLOBAL_WORKERS
        cml.GLOBAL_WORKERS = n_workers

        def restore():
            cml.GLOBAL_WORKERS = original
        return restore
    except (ImportError, AttributeError):
        pass

    # Fallback: transformers v4 env var API
    old_enable = os.environ.get("HF_ENABLE_PARALLEL_LOADING")
    old_workers = os.environ.get("HF_PARALLEL_LOADING_WORKERS")
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "1"
    os.environ["HF_PARALLEL_LOADING_WORKERS"] = str(n_workers)

    def restore():
        if old_enable is None:
            os.environ.pop("HF_ENABLE_PARALLEL_LOADING", None)
        else:
            os.environ["HF_ENABLE_PARALLEL_LOADING"] = old_enable
        if old_workers is None:
            os.environ.pop("HF_PARALLEL_LOADING_WORKERS", None)
        else:
            os.environ["HF_PARALLEL_LOADING_WORKERS"] = old_workers
    return restore


def run_hf(model_id: str, gpu_ids: list[int],
           workers: int = 4, revision: str = "main") -> TimingResult:
    """Load via LanguageModel with load_format='from_pretrained'."""
    from nnsight import LanguageModel

    max_memory = build_max_memory(gpu_ids)
    restore = _patch_hf_workers(workers)

    try:
        with PeakMemMonitor() as mem:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            model = LanguageModel(
                model_id,
                load_format="from_pretrained",
                device_map="auto",
                max_memory=max_memory,
                revision=revision,
                dispatch=True,
            )

            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

        peak_gpu = get_gpu_mem_allocated_mb(gpu_ids)
        unload_model(model)
    finally:
        restore()

    return TimingResult(
        experiment="hf",
        config={"workers": workers},
        wall_time_s=wall,
        peak_gpu_mem_mb=peak_gpu,
        peak_rss_mb=mem.peak_rss_mb,
        peak_private_mb=mem.peak_private_mb,
    )


def run_runai_lazy(model_id: str, gpu_ids: list[int],
                   concurrency: int = 16,
                   revision: str = "main") -> TimingResult:
    """Load via run:ai — lazy shard-by-shard streaming (new default)."""
    from nnsight import LanguageModel

    max_memory = build_max_memory(gpu_ids)
    restore = _patch_hf_workers(concurrency)

    try:
        with PeakMemMonitor() as mem:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            model = LanguageModel(
                model_id,
                device_map="auto",
                max_memory=max_memory,
                concurrency=concurrency,
                revision=revision,
                dispatch=True,
            )

            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

        peak_gpu = get_gpu_mem_allocated_mb(gpu_ids)
        unload_model(model)
    finally:
        restore()

    return TimingResult(
        experiment="runai_lazy",
        config={"concurrency": concurrency},
        wall_time_s=wall,
        peak_gpu_mem_mb=peak_gpu,
        peak_rss_mb=mem.peak_rss_mb,
        peak_private_mb=mem.peak_private_mb,
    )


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def _load_and_get_logits(model_id, gpu_ids, revision, prompt, **extra_kwargs):
    """Load model, run one forward pass, return (logits, decoded_token, token_id)."""
    from nnsight import LanguageModel

    max_memory = build_max_memory(gpu_ids)
    model = LanguageModel(
        model_id, device_map="auto", max_memory=max_memory,
        revision=revision, dispatch=True, **extra_kwargs,
    )
    with model.trace(prompt):
        logits = model.lm_head.output.save()
    token_id = logits[0, -1].argmax(dim=-1).item()
    decoded = model.tokenizer.decode(token_id)
    logits_cpu = logits.cpu().float()
    unload_model(model)
    return logits_cpu, decoded, token_id


def verify_outputs(model_id: str, gpu_ids: list[int],
                   experiments: list[str],
                   revision: str = "main") -> bool:
    """Load with each selected path, run same prompt, compare logits.

    Uses 'hf' as the ground truth baseline when available, otherwise
    compares all against the first experiment.
    """
    if len(experiments) < 2:
        print("  Skipping verification (need at least 2 experiments)")
        return True

    prompt = "The Eiffel Tower is in the city of"
    print(f"\n  Verifying output correctness (prompt: {prompt!r})...")

    # Map experiment name → extra kwargs for LanguageModel
    load_kwargs = {
        "hf":          {"load_format": "from_pretrained"},
        "runai_lazy":  {},  # default path
    }

    # Decide baseline order: prefer hf first
    ordered = sorted(experiments, key=lambda e: (e != "hf", e))
    results = {}
    for exp in ordered:
        print(f"    Loading {exp}...")
        results[exp] = _load_and_get_logits(
            model_id, gpu_ids, revision, prompt, **load_kwargs[exp],
        )

    baseline_name = ordered[0]
    baseline_logits, baseline_decoded, baseline_token = results[baseline_name]
    print(f"    Baseline ({baseline_name}): {baseline_decoded!r} (id={baseline_token})")

    all_ok = True
    for exp in ordered[1:]:
        logits, decoded, token_id = results[exp]
        match = torch.allclose(baseline_logits, logits, atol=1e-4)
        max_diff = (baseline_logits - logits).abs().max().item()
        token_ok = token_id == baseline_token
        print(f"    {exp}: {decoded!r} (id={token_id}) | "
              f"logits {'MATCH' if match else 'MISMATCH'} "
              f"(max diff: {max_diff:.2e}) | "
              f"token {'MATCH' if token_ok else 'MISMATCH'}")
        if not match or not token_ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = ["hf", "runai_lazy"]

EXPERIMENT_CONFIGS = {
    "hf":          [("hf",          {"workers": w})     for w in [1, 2, 4, 8, 16, 32]],
    "runai_lazy":  [("runai_lazy",  {"concurrency": c}) for c in [1, 2, 4, 8, 16, 32]],
}

_RUNNERS = {
    "hf":          lambda mid, gids, cfg, rev: run_hf(
                       mid, gids, workers=cfg.get("workers", 4), revision=rev),
    "runai_lazy":  lambda mid, gids, cfg, rev: run_runai_lazy(
                       mid, gids, concurrency=cfg.get("concurrency", 16), revision=rev),
}


def run_single_config(exp_name, config, model_id, gpu_ids, revision):
    runner = _RUNNERS.get(exp_name)
    if runner is None:
        raise ValueError(f"Unknown experiment: {exp_name}")
    return runner(model_id, gpu_ids, config, revision)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark nnsight model loading: hf vs runai_lazy"
    )
    parser.add_argument("--model", default="openai-community/gpt2",
                        help="HuggingFace model ID")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs (default: all)")
    parser.add_argument("--experiments", nargs="+",
                        default=ALL_EXPERIMENTS,
                        choices=ALL_EXPERIMENTS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sudo-drop-caches", action="store_true")
    parser.add_argument("--no-drop-caches", action="store_true")
    parser.add_argument("--junk-drop-caches", action="store_true",
                        help="Use junk file to pressure page cache (unreliable)")
    parser.add_argument("--junk-size-gb", type=int, default=JUNK_FILE_SIZE_GB)
    parser.add_argument("--warmup", action="store_true",
                        help="Read model shards into page cache before timing")
    parser.add_argument("--disk-device", default="nvme0n1",
                        help="Block device for disk read stats (default: nvme0n1)")
    parser.add_argument("--net-iface", default="hsn0",
                        help="Network interface for RX/TX stats (default: hsn0)")
    parser.add_argument("--no-verify", action="store_true", default=False,
                        help="Skip output correctness verification")
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    # Check run:ai availability
    try:
        from runai_model_streamer import SafetensorsStreamer  # noqa: F401
        has_runai = True
    except ImportError:
        has_runai = False

    runai_exps = {"runai_lazy"}
    if not has_runai:
        skipped = [e for e in args.experiments if e in runai_exps]
        if skipped:
            print(f"WARNING: runai-model-streamer not installed. "
                  f"Skipping {skipped}.")
        args.experiments = [e for e in args.experiments if e not in runai_exps]
    if not args.experiments:
        print("ERROR: No valid experiments. Exiting.")
        sys.exit(1)

    # Ensure model is downloaded before any experiments or cache operations
    from huggingface_hub import snapshot_download
    print(f"Downloading model (if needed): {args.model}")
    snapshot_download(args.model, revision=args.revision)

    # Resolve model size
    model_size_gb = resolve_model_size_gb(args.model, args.revision)

    print(f"Model:       {args.model}")
    print(f"Model size:  {model_size_gb:.1f} GB")
    print(f"GPUs:        {gpu_ids}")
    print(f"Repeats:     {args.repeats}")
    print(f"Experiments: {args.experiments}")
    print(f"run:ai:      {'available' if has_runai else 'NOT installed'}")

    # Cache invalidation setup
    junk_file = None
    if args.sudo_drop_caches:
        cache_mode = "sudo_drop_caches"
        print("Cache mode:  sudo drop_caches")
        try:
            subprocess.run(["sudo", "-n", "true"], check=True, timeout=5,
                           capture_output=True)
        except Exception:
            print("ERROR: --sudo-drop-caches requires passwordless sudo.")
            sys.exit(1)
    elif args.no_drop_caches:
        cache_mode = "warm" if args.warmup else "disabled"
        print(f"Cache mode:  {cache_mode}")
    elif args.junk_drop_caches:
        cache_mode = "junk_file"
        print(f"Cache mode:  junk file ({args.junk_size_gb} GB on /tmp)")
        junk_file = JUNK_FILE_PATH
        create_junk_file(junk_file, args.junk_size_gb)
    else:
        cache_mode = "cold"
        print("Cache mode:  fadvise (FADV_DONTNEED on model shards)")

    def drop_caches():
        if args.no_drop_caches:
            return
        if args.sudo_drop_caches:
            sudo_drop_caches()
        elif args.junk_drop_caches:
            invalidate_via_junk_file(junk_file)
        else:
            evict_model_pages(args.model, args.revision)

    # Correctness verification
    if not args.no_verify and len(args.experiments) > 1:
        ok = verify_outputs(args.model, gpu_ids, args.experiments, args.revision)
        if not ok:
            print("\n  WARNING: Output mismatch between loading paths!")
        else:
            print("  PASS: All outputs match.")

    # Warmup: populate page cache before timed runs
    if args.warmup:
        warm_model_pages(args.model, args.revision)

    # Run experiments
    all_results = []
    for exp_name in args.experiments:
        # Reset cache state between experiment groups so that e.g.
        # runai_eager's 61 GB CPU footprint doesn't warm the cache
        # for the next group (runai_lazy).
        gc.collect()
        drop_caches()
        if args.warmup:
            warm_model_pages(args.model, args.revision)

        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_name}")
        print(f"{'=' * 60}")

        for rep in range(args.repeats):
            if args.repeats > 1:
                print(f"\n--- Repeat {rep + 1}/{args.repeats} ---")

            for _, config in EXPERIMENT_CONFIGS[exp_name]:
                drop_caches()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                try:
                    disk_before = _read_diskstats(args.disk_device)
                    net_before = _read_iface_bytes(args.net_iface)

                    result = run_single_config(
                        exp_name, config, args.model, gpu_ids,
                        args.revision,
                    )

                    disk_after = _read_diskstats(args.disk_device)
                    net_after = _read_iface_bytes(args.net_iface)

                    result.config["repeat"] = rep

                    if disk_before is not None and disk_after is not None:
                        result.disk_read_gib = disk_after - disk_before
                    if net_before is not None and net_after is not None:
                        result.net_rx_gib = net_after[0] - net_before[0]
                        result.net_tx_gib = net_after[1] - net_before[1]

                    bw = model_size_gb / result.wall_time_s if result.wall_time_s > 0 else 0
                    print(f"\n  [OK] {result.experiment} | config={result.config}")
                    print(f"    wall_time:    {result.wall_time_s:.2f}s")
                    print(f"    bw:           {bw:.2f} GB/s")
                    print(f"    peak_gpu_mem: {result.peak_gpu_mem_mb:.0f} MB")
                    print(f"    peak_rss:     {result.peak_rss_mb:.0f} MB")
                    print(f"    peak_private: {result.peak_private_mb:.0f} MB")
                    if result.disk_read_gib is not None:
                        print(f"    disk_read:    {result.disk_read_gib:.3f} GiB")
                    if result.net_rx_gib is not None:
                        print(f"    net_rx:       {result.net_rx_gib:.3f} GiB")

                    all_results.append(result)
                except Exception as e:
                    print(f"\n  [ERROR] {exp_name} config={config}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append(TimingResult(
                        experiment=exp_name, config={**config, "repeat": rep},
                        wall_time_s=0, error=str(e),
                    ))

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    hdr = (f"{'Experiment':<14} {'Config':<28} {'Wall (s)':<10} "
           f"{'BW (GB/s)':<10} {'GPU (MB)':<10} {'RSS (MB)':<10} "
           f"{'Private (MB)':<12} {'Disk (GiB)':<10}")
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        if r.error:
            continue
        config_str = json.dumps(r.config, default=str)
        if len(config_str) > 26:
            config_str = config_str[:23] + "..."
        bw = model_size_gb / r.wall_time_s if r.wall_time_s > 0 else 0
        disk_str = f"{r.disk_read_gib:.3f}" if r.disk_read_gib is not None else "n/a"
        print(f"{r.experiment:<14} {config_str:<28} {r.wall_time_s:<10.2f} "
              f"{bw:<10.2f} {r.peak_gpu_mem_mb:<10.0f} {r.peak_rss_mb:<10.0f} "
              f"{r.peak_private_mb:<12.0f} {disk_str:<10}")

    # JSON output
    if args.output:
        out = [{
            "experiment": r.experiment, "config": r.config,
            "wall_time_s": r.wall_time_s,
            "peak_gpu_mem_mb": r.peak_gpu_mem_mb,
            "peak_rss_mb": r.peak_rss_mb,
            "peak_private_mb": r.peak_private_mb,
            "disk_read_gib": r.disk_read_gib,
            "net_rx_gib": r.net_rx_gib,
            "net_tx_gib": r.net_tx_gib,
            "error": r.error,
        } for r in all_results]
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model, "model_size_gb": model_size_gb,
                "gpus": gpu_ids, "cache_mode": cache_mode,
                "results": out,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

"""Benchmark: nnsight LanguageModel loading — run:ai streamer vs from_pretrained.

Compares wall-clock time, peak GPU/CPU memory, and output correctness for
loading HuggingFace models through nnsight's LanguageModel interface using
the default run:ai SafetensorsStreamer path vs explicit from_pretrained.

Page cache invalidation between experiments:
  By default, creates a large junk file on /tmp and reads it between runs
  to pressure model shard pages out of the kernel page cache.
  With --sudo-drop-caches, uses 'echo 3 > /proc/sys/vm/drop_caches' instead.
  With --no-drop-caches, skips invalidation entirely (warm-cache runs).

Usage:
  # Quick warm-cache test with GPT-2
  python benchmark_loading.py --model openai-community/gpt2 --gpus 1 --no-drop-caches

  # Full cold-cache benchmark
  python benchmark_loading.py --model meta-llama/Llama-3.1-8B --gpus 0,1 --repeats 3

  # Specific experiments
  python benchmark_loading.py --model Qwen/Qwen2.5-7B-Instruct --experiments hf runai
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    peak_cpu_mem_mb: float = 0.0
    error: Optional[str] = None


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
JUNK_FILE_SIZE_GB = 256


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


def run_hf(model_id: str, gpu_ids: list[int], dtype: torch.dtype,
           workers: int = 4, revision: str = "main") -> TimingResult:
    """Load via LanguageModel with load_format='from_pretrained'.

    Args:
        workers: Number of worker threads for HF shard loading.
    """
    from nnsight import LanguageModel

    max_memory = build_max_memory(gpu_ids)
    restore = _patch_hf_workers(workers)

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        model = LanguageModel(
            model_id,
            load_format="from_pretrained",
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=dtype,
            revision=revision,
            dispatch=True,
        )

        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        peak_gpu = get_gpu_mem_allocated_mb(gpu_ids)
        peak_cpu = get_process_rss_mb()
        unload_model(model)
    finally:
        restore()

    return TimingResult(
        experiment="hf",
        config={"workers": workers},
        wall_time_s=wall,
        peak_gpu_mem_mb=peak_gpu,
        peak_cpu_mem_mb=peak_cpu,
    )


def run_runai(model_id: str, gpu_ids: list[int], dtype: torch.dtype,
              concurrency: int = 16, revision: str = "main") -> TimingResult:
    """Load via LanguageModel with run:ai streaming + from_pretrained."""
    from nnsight import LanguageModel

    max_memory = build_max_memory(gpu_ids)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model = LanguageModel(
        model_id,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=dtype,
        concurrency=concurrency,
        revision=revision,
        dispatch=True,
    )

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    peak_gpu = get_gpu_mem_allocated_mb(gpu_ids)
    peak_cpu = get_process_rss_mb()
    unload_model(model)

    return TimingResult(
        experiment="runai",
        config={"concurrency": concurrency},
        wall_time_s=wall,
        peak_gpu_mem_mb=peak_gpu,
        peak_cpu_mem_mb=peak_cpu,
    )


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def verify_outputs(model_id: str, gpu_ids: list[int], dtype: torch.dtype,
                   revision: str = "main") -> bool:
    """Load with both paths, run same prompt, compare logits."""
    from nnsight import LanguageModel

    max_memory = build_max_memory(gpu_ids)
    prompt = "The Eiffel Tower is in the city of"

    print("\n  Verifying output correctness...")

    # HF path
    model_hf = LanguageModel(
        model_id, load_format="from_pretrained",
        device_map="auto", max_memory=max_memory,
        torch_dtype=dtype, revision=revision, dispatch=True,
    )
    with model_hf.trace(prompt):
        logits_hf = model_hf.lm_head.output.save()
    token_hf = logits_hf[0, -1].argmax(dim=-1).item()
    decoded_hf = model_hf.tokenizer.decode(token_hf)
    unload_model(model_hf)

    # run:ai path
    model_runai = LanguageModel(
        model_id, device_map="auto", max_memory=max_memory,
        torch_dtype=dtype, revision=revision, dispatch=True,
    )
    with model_runai.trace(prompt):
        logits_runai = model_runai.lm_head.output.save()
    token_runai = logits_runai[0, -1].argmax(dim=-1).item()
    decoded_runai = model_runai.tokenizer.decode(token_runai)
    unload_model(model_runai)

    # Compare
    match = torch.allclose(
        logits_hf.cpu().float(), logits_runai.cpu().float(), atol=1e-4
    )
    max_diff = (logits_hf.cpu().float() - logits_runai.cpu().float()).abs().max().item()

    print(f"    HF next token:    {decoded_hf!r} (id={token_hf})")
    print(f"    run:ai next token: {decoded_runai!r} (id={token_runai})")
    print(f"    Logits match:     {match} (max diff: {max_diff:.2e})")
    print(f"    Token match:      {token_hf == token_runai}")

    return match and token_hf == token_runai


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS = {
    "hf": [("hf", {"workers": w}) for w in [4, 8, 16]],
    "runai": [("runai", {"concurrency": c}) for c in [4, 8, 16]],
}


def run_single_config(exp_name, config, model_id, gpu_ids, dtype, revision):
    if exp_name == "hf":
        return run_hf(
            model_id, gpu_ids, dtype,
            workers=config.get("workers", 4),
            revision=revision,
        )
    elif exp_name == "runai":
        return run_runai(
            model_id, gpu_ids, dtype,
            concurrency=config.get("concurrency", 16),
            revision=revision,
        )
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark nnsight model loading: run:ai vs from_pretrained"
    )
    parser.add_argument("--model", default="openai-community/gpt2",
                        help="HuggingFace model ID")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs (default: all)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--experiments", nargs="+",
                        default=list(EXPERIMENT_CONFIGS.keys()),
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sudo-drop-caches", action="store_true")
    parser.add_argument("--no-drop-caches", action="store_true")
    parser.add_argument("--junk-size-gb", type=int, default=JUNK_FILE_SIZE_GB)
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip output correctness verification")
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]

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

    if "runai" in args.experiments and not has_runai:
        print("WARNING: runai-model-streamer not installed. Skipping 'runai' experiment.")
        args.experiments = [e for e in args.experiments if e != "runai"]
    if not args.experiments:
        print("ERROR: No valid experiments. Exiting.")
        sys.exit(1)

    # Resolve model size
    model_size_gb = resolve_model_size_gb(args.model, args.revision)

    print(f"Model:       {args.model}")
    print(f"Model size:  {model_size_gb:.1f} GB")
    print(f"GPUs:        {gpu_ids}")
    print(f"Dtype:       {args.dtype}")
    print(f"Repeats:     {args.repeats}")
    print(f"Experiments: {args.experiments}")
    print(f"run:ai:      {'available' if has_runai else 'NOT installed'}")

    # Cache invalidation setup
    junk_file = None
    if args.sudo_drop_caches:
        print("Cache mode:  sudo drop_caches")
        try:
            subprocess.run(["sudo", "-n", "true"], check=True, timeout=5,
                           capture_output=True)
        except Exception:
            print("ERROR: --sudo-drop-caches requires passwordless sudo.")
            sys.exit(1)
    elif args.no_drop_caches:
        print("Cache mode:  disabled")
    else:
        print(f"Cache mode:  junk file ({args.junk_size_gb} GB on /tmp)")
        junk_file = JUNK_FILE_PATH
        create_junk_file(junk_file, args.junk_size_gb)

    def drop_caches():
        if args.no_drop_caches:
            return
        if args.sudo_drop_caches:
            sudo_drop_caches()
        else:
            invalidate_via_junk_file(junk_file)

    # Correctness verification
    if not args.no_verify and has_runai and len(args.experiments) > 1:
        ok = verify_outputs(args.model, gpu_ids, dtype, args.revision)
        if not ok:
            print("\n  WARNING: Output mismatch between loading paths!")
        else:
            print("  PASS: Outputs match.")

    # Run experiments
    all_results = []
    for exp_name in args.experiments:
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
                    result = run_single_config(
                        exp_name, config, args.model, gpu_ids,
                        dtype, args.revision,
                    )
                    result.config["repeat"] = rep
                    bw = model_size_gb / result.wall_time_s if result.wall_time_s > 0 else 0
                    print(f"\n  [OK] {result.experiment} | config={result.config}")
                    print(f"    wall_time:    {result.wall_time_s:.2f}s")
                    print(f"    bw:           {bw:.2f} GB/s")
                    print(f"    peak_gpu_mem: {result.peak_gpu_mem_mb:.0f} MB")
                    print(f"    peak_cpu_mem: {result.peak_cpu_mem_mb:.0f} MB")
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
    print(f"{'Experiment':<12} {'Config':<28} {'Wall (s)':<12} {'BW (GB/s)':<12} {'GPU (MB)':<12}")
    print("-" * 76)
    for r in all_results:
        if not r.error:
            config_str = json.dumps(r.config, default=str)
            if len(config_str) > 26:
                config_str = config_str[:23] + "..."
            bw = model_size_gb / r.wall_time_s if r.wall_time_s > 0 else 0
            print(f"{r.experiment:<12} {config_str:<28} {r.wall_time_s:<12.2f} {bw:<12.2f} {r.peak_gpu_mem_mb:<12.0f}")

    # JSON output
    if args.output:
        out = [{
            "experiment": r.experiment, "config": r.config,
            "wall_time_s": r.wall_time_s,
            "peak_gpu_mem_mb": r.peak_gpu_mem_mb,
            "peak_cpu_mem_mb": r.peak_cpu_mem_mb,
            "error": r.error,
        } for r in all_results]
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model, "model_size_gb": model_size_gb,
                "gpus": gpu_ids, "dtype": args.dtype, "results": out,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

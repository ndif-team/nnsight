"""Profile loading breakdown for each experiment at a fixed thread count.

Uses LanguageModel directly (identical to benchmark) and monkey-patches
to capture phase timing and cache stats.

Usage:
  python profile_loading.py --model Qwen/Qwen3-32B --gpus 0,1 --threads 8
"""

import gc
import os
import time
from pathlib import Path

import torch


# -- helpers ------------------------------------------------------------------

def evict_model_pages(model_id, revision="main"):
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(model_id, revision=revision, local_files_only=True)
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def get_per_gpu_mb(gpu_ids):
    return {i: torch.cuda.memory_allocated(i) / 1024**2 for i in gpu_ids}


def build_max_memory_balanced(gpu_ids, model_size_bytes):
    n = len(gpu_ids)
    mm = {}
    for i in range(torch.cuda.device_count()):
        if i in gpu_ids:
            phys = torch.cuda.get_device_properties(i).total_memory
            per = int(model_size_bytes / n * 1.15)
            mm[i] = min(per, int(phys * 0.9))
        else:
            mm[i] = 0
    return mm


def resolve_model_size_bytes(model_id, revision="main"):
    from huggingface_hub import snapshot_download
    d = snapshot_download(model_id, revision=revision, local_files_only=True)
    paths = sorted(Path(d).glob("*.safetensors"))
    return sum(p.stat().st_size for p in paths)


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def patch_hf_workers(n):
    try:
        import transformers.core_model_loading as cml
        orig = cml.GLOBAL_WORKERS
        cml.GLOBAL_WORKERS = n
        return lambda: setattr(cml, "GLOBAL_WORKERS", orig)
    except (ImportError, AttributeError):
        pass
    old = os.environ.get("HF_PARALLEL_LOADING_WORKERS")
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "1"
    os.environ["HF_PARALLEL_LOADING_WORKERS"] = str(n)
    def restore():
        if old is None:
            os.environ.pop("HF_PARALLEL_LOADING_WORKERS", None)
        else:
            os.environ["HF_PARALLEL_LOADING_WORKERS"] = old
    return restore


def _read_diskstats(device="md0"):
    stat_path = f"/sys/class/block/{device}/stat"
    try:
        with open(stat_path) as f:
            return int(f.read().split()[2]) * 512 / 1024**3
    except Exception:
        return None


# -- profiling via monkey-patch -----------------------------------------------

def profile_experiment(model_id, gpu_ids, exp_name, threads, max_memory,
                       revision="main"):
    """Profile a single experiment using LanguageModel with timing hooks."""
    from nnsight import LanguageModel
    import nnsight.modeling.transformers as tm_mod
    import nnsight.modeling.loader as loader_mod

    restore = patch_hf_workers(threads)

    # Timing captures
    timings = {}
    captured_cache = {}

    # Monkey-patch _resolve_device_map to time it
    orig_resolve_dm = tm_mod.TransformersModel._resolve_device_map

    def timed_resolve_dm(self, *a, **kw):
        print(f"  [patch] _resolve_device_map called")
        t0 = time.perf_counter()
        result = orig_resolve_dm(self, *a, **kw)
        timings["resolve_device_map"] = time.perf_counter() - t0
        print(f"  [patch] _resolve_device_map took {timings['resolve_device_map']:.2f}s")
        return result

    tm_mod.TransformersModel._resolve_device_map = timed_resolve_dm

    # Monkey-patch build_lazy_state_dict to time it and capture cache
    orig_build = loader_mod.build_lazy_state_dict

    def timed_build(*a, **kw):
        print(f"  [patch] build_lazy_state_dict called")
        t0 = time.perf_counter()
        result = orig_build(*a, **kw)
        timings["build_state_dict"] = time.perf_counter() - t0
        for v in result.values():
            captured_cache["cache"] = v._cache
            break
        print(f"  [patch] build_lazy_state_dict took {timings['build_state_dict']:.3f}s")
        return result

    loader_mod.build_lazy_state_dict = timed_build

    # Build kwargs
    extra = {}
    if exp_name == "hf":
        extra = {"load_format": "from_pretrained"}
    elif exp_name == "runai_stream":
        extra = {"concurrency": threads, "gpu_direct": False}
    elif exp_name == "runai_gpu_direct":
        extra = {"concurrency": threads}
    elif exp_name == "runai_gpu_direct_lazy":
        extra = {"concurrency": threads, "lazy": True}

    try:
        evict_model_pages(model_id, revision)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        disk_before = _read_diskstats()
        torch.cuda.synchronize()
        t_total = time.perf_counter()

        model = LanguageModel(
            model_id,
            device_map="auto",
            max_memory=max_memory,
            revision=revision,
            dispatch=True,
            **extra,
        )

        torch.cuda.synchronize()
        t_total = time.perf_counter() - t_total
        disk_after = _read_diskstats()

        per_gpu = get_per_gpu_mb(gpu_ids)
        unload(model)
    finally:
        tm_mod.TransformersModel._resolve_device_map = orig_resolve_dm
        loader_mod.build_lazy_state_dict = orig_build
        restore()

    disk_gib = None
    if disk_before is not None and disk_after is not None:
        disk_gib = disk_after - disk_before

    result = {
        "experiment": exp_name,
        "total_s": t_total,
        "per_gpu_mb": per_gpu,
        "disk_read_gib": disk_gib,
        "resolve_device_map_s": timings.get("resolve_device_map"),
        "build_state_dict_s": timings.get("build_state_dict"),
    }

    cache = captured_cache.get("cache")
    if cache:
        result["cache_stats"] = {
            "shard_wall_s": cache.stats_shard_wall_s,
            "io_wait_s": cache.stats_io_wait_s,
            "clone_s": cache.stats_clone_s,
            "gpu_copy_s": cache.stats_gpu_copy_s,
            "consumer_wait_s": cache.stats_consumer_wait_s,
            "pop_count": cache.stats_pop_count,
        }

    return result


# -- main --------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    model_size = resolve_model_size_bytes(args.model, args.revision)
    max_memory = build_max_memory_balanced(gpu_ids, model_size)

    print(f"Model:      {args.model}")
    print(f"Size:       {model_size / 1024**3:.1f} GB")
    print(f"GPUs:       {gpu_ids}")
    print(f"Threads:    {args.threads}")
    print(f"max_memory: { {k: f'{v/1024**3:.1f}GB' for k,v in max_memory.items() if v > 0} }")

    experiments = ["hf", "runai_stream", "runai_gpu_direct", "runai_gpu_direct_lazy"]

    results = []
    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Profiling: {exp_name}")
        print(f"{'='*60}")

        r = profile_experiment(args.model, gpu_ids, exp_name, args.threads,
                               max_memory, args.revision)
        results.append(r)

        # Print detailed results
        print(f"\n  Total:              {r['total_s']:.2f}s")
        if r.get("disk_read_gib") is not None:
            print(f"  Disk read:          {r['disk_read_gib']:.2f} GiB")
        if r.get("resolve_device_map_s") is not None:
            print(f"  resolve_device_map: {r['resolve_device_map_s']:.2f}s")
        if r.get("build_state_dict_s") is not None:
            print(f"  build_state_dict:   {r['build_state_dict_s']:.3f}s")

        cs = r.get("cache_stats")
        if cs:
            # from_pretrained time = total - resolve_dm - build_sd
            dm = r.get("resolve_device_map_s", 0)
            sd = r.get("build_state_dict_s", 0)
            fp = r["total_s"] - dm - sd
            print(f"  from_pretrained:    {fp:.2f}s  (= total - dm - sd)")
            print(f"  --- cache breakdown (cumulative across loader threads) ---")
            print(f"    shard_wall:     {cs['shard_wall_s']:.2f}s  (total time inside streamer)")
            print(f"    io_wait:        {cs['io_wait_s']:.2f}s  (blocked waiting for disk)")
            print(f"    gpu_copy:       {cs['gpu_copy_s']:.2f}s  (buffer → GPU .to())")
            print(f"    clone:          {cs['clone_s']:.2f}s  (buffer → CPU clone)")
            overhead = cs['shard_wall_s'] - cs['io_wait_s'] - cs['gpu_copy_s'] - cs['clone_s']
            print(f"    lock+notify:    {overhead:.2f}s")
            print(f"    consumer_wait:  {cs['consumer_wait_s']:.2f}s  (HF workers blocked on cache)")
            print(f"    tensors served: {cs['pop_count']}")

        per = r["per_gpu_mb"]
        for gid, mb in sorted(per.items()):
            print(f"  GPU {gid}: {mb:.0f} MB")

    # Summary table
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    hdr = (f"{'Experiment':<22} {'Total':>6} {'Disk':>6} {'DM':>5} {'SD':>5} "
           f"{'FP':>5} {'IO':>5} {'GPU':>5} {'Clone':>5} {'CWait':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        disk_str = f"{r['disk_read_gib']:.1f}" if r.get('disk_read_gib') is not None else "n/a"
        dm = r.get('resolve_device_map_s', 0) or 0
        sd = r.get('build_state_dict_s', 0) or 0
        fp = r['total_s'] - dm - sd

        cs = r.get("cache_stats")
        if cs:
            print(f"{r['experiment']:<22} {r['total_s']:>6.1f} {disk_str:>6} "
                  f"{dm:>5.1f} {sd:>5.2f} {fp:>5.1f} "
                  f"{cs['io_wait_s']:>5.1f} {cs['gpu_copy_s']:>5.1f} "
                  f"{cs['clone_s']:>5.1f} {cs['consumer_wait_s']:>5.1f}")
        else:
            print(f"{r['experiment']:<22} {r['total_s']:>6.1f} {disk_str:>6} "
                  f"{'':>5} {'':>5} {fp:>5.1f}   "
                  f"(from_pretrained handles everything)")

    print(f"\nLegend: DM=resolve_device_map, SD=build_state_dict, "
          f"FP=from_pretrained,")
    print(f"        IO=disk_io_wait, GPU=gpu_copy, CWait=consumer_wait")


if __name__ == "__main__":
    main()

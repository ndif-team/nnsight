#!/usr/bin/env python3
"""Raw cold-cache read throughput of a cached model's safetensors shards.

Isolates the filesystem from the loaders: evict the shard pages, then read every
shard with N threads using large sequential read() calls (16 MB), and report
GB/s. This is the ceiling a loader can reach on that storage with N readers.

    python raw_read.py --model Qwen/Qwen3-32B --threads 1 4 8 16 --output raw.json
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download


def evict(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def read_file(path, buf=16 * 1024 * 1024):
    with open(path, "rb", buffering=0) as f:
        while f.read(buf):
            pass


def mmap_file(path, chunk=64 * 1024 * 1024):
    """Read a file through mmap page faults, the way safetensors' mmap backend is
    consumed: map it, view it as a tensor, and copy it out in tensor-sized pieces.
    The copy (torch clone) releases the GIL, so threads fault pages in parallel."""
    import mmap

    import torch

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    try:
        view = torch.frombuffer(mm, dtype=torch.uint8)
        for off in range(0, view.numel(), chunk):
            view[off:off + chunk].clone()
        del view
    finally:
        try:
            mm.close()
        except BufferError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--mmap-threads", type=int, nargs="*", default=[1, 4, 16],
                    help="thread counts for the mmap page-fault pass (empty = skip)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    model_dir = snapshot_download(args.model, local_files_only=True)
    paths = [str(p) for p in sorted(Path(model_dir).glob("*.safetensors"))]
    total_gb = sum(os.path.getsize(p) for p in paths) / 1024**3
    print(f"raw read: {args.model}, {len(paths)} shards, {total_gb:.1f} GB")
    results = []
    for n in args.threads:
        evict(paths)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(read_file, paths))
        dt = time.perf_counter() - t0
        print(f"  read() threads={n:<3d} {dt:7.2f}s  {total_gb / dt:6.2f} GB/s", flush=True)
        results.append({"mode": "read", "threads": n, "wall_time_s": dt, "bw_gbs": total_gb / dt})
    import warnings
    warnings.filterwarnings("ignore", message=".*not writable.*")
    if args.mmap_threads:
        import torch
        default_threads = torch.get_num_threads()
        # Two access patterns over the same mapping:
        #   mmap-seq     each reader copies its file front to back with ONE thread,
        #                so page faults arrive in order and readahead can work
        #                (what one HF worker does to one tensor).
        #   mmap-scatter torch splits every 64 MB copy across its intra-op threads,
        #                so faults land at many offsets at once and readahead
        #                never sees a sequential stream.
        for mode, torch_threads in (("mmap-seq", 1), ("mmap-scatter", default_threads)):
            torch.set_num_threads(torch_threads)
            for n in args.mmap_threads:
                evict(paths)
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n) as pool:
                    list(pool.map(mmap_file, paths))
                dt = time.perf_counter() - t0
                print(f"  {mode:<12s} threads={n:<3d} {dt:7.2f}s  {total_gb / dt:6.2f} GB/s "
                      f"(torch copy threads={torch_threads})", flush=True)
                results.append({"mode": mode, "threads": n, "torch_threads": torch_threads,
                                "wall_time_s": dt, "bw_gbs": total_gb / dt})
        torch.set_num_threads(default_threads)
    evict(paths)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "model_size_gb": total_gb, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()

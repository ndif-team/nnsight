"""Benchmark file copy throughput from Lustre HF cache to /tmp.

Compares shutil.copy2 (sendfile), raw read/write, and concurrent copies
to understand what I/O pattern works best on Lustre.

Usage:
    python bench_copy.py --model meta-llama/Llama-3.1-8B
    python bench_copy.py --model Qwen/Qwen2.5-32B-Instruct --workers 1 2 4 8 16 32
"""

import argparse
import os
import shutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_shard_paths(repo_id: str) -> list[str]:
    model_dir = snapshot_download(repo_id, local_files_only=True)
    paths = sorted(Path(model_dir).glob("*.safetensors"))
    if not paths:
        raise ValueError(f"No .safetensors files in {model_dir}")
    return [str(p) for p in paths]


_JUNK_DIR = "/tmp/_bench_junk"
_JUNK_N_FILES = 16


class Progress:
    """Thread-safe byte counter with periodic status printing."""

    def __init__(self, total_bytes: int, label: str = ""):
        self.total = total_bytes
        self.label = label
        self.done = 0
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()

    def update(self, n_bytes: int):
        with self._lock:
            self.done += n_bytes
            elapsed = time.perf_counter() - self._t0
            pct = self.done / self.total * 100 if self.total else 0
            bw = (self.done / (1024**3)) / elapsed if elapsed > 0 else 0
            sys.stderr.write(
                f"\r  {self.label} {pct:5.1f}%  "
                f"{self.done / (1024**3):.1f}/{self.total / (1024**3):.1f} GB  "
                f"{bw:.2f} GB/s"
            )
            sys.stderr.flush()

    def finish(self):
        elapsed = time.perf_counter() - self._t0
        sys.stderr.write("\n")
        sys.stderr.flush()
        return elapsed


def prepare_junk_files(junk_gb: float):
    """Write junk files to /tmp once. Reused by evict_page_cache().

    Uses concurrent writers to saturate the local NVMe.
    """
    if junk_gb <= 0:
        return

    if os.path.isdir(_JUNK_DIR):
        existing = sum(
            os.path.getsize(os.path.join(_JUNK_DIR, f))
            for f in os.listdir(_JUNK_DIR) if f.startswith("junk_")
        )
        if existing >= junk_gb * 0.95 * (1024**3):
            print(f"  [junk] reusing existing {existing / (1024**3):.1f} GB junk files")
            return

    os.makedirs(_JUNK_DIR, exist_ok=True)
    chunk = os.urandom(4 * 1024 * 1024)  # 4 MB
    chunks_per_file = max(1, int(junk_gb * 1024 / 4) // _JUNK_N_FILES)

    total_bytes = chunks_per_file * _JUNK_N_FILES * 4 * 1024 * 1024
    prog = Progress(total_bytes, "[junk write]")

    def write_one(idx):
        path = os.path.join(_JUNK_DIR, f"junk_{idx}")
        with open(path, "wb") as f:
            for _ in range(chunks_per_file):
                f.write(chunk)
                prog.update(4 * 1024 * 1024)

    with ThreadPoolExecutor(max_workers=_JUNK_N_FILES) as pool:
        futs = [pool.submit(write_one, i) for i in range(_JUNK_N_FILES)]
        for f in as_completed(futs):
            f.result()
    prog.finish()


def evict_page_cache(junk_gb: float = 0):
    """Evict page cache by reading pre-written junk files from local SSD.

    Reads junk files concurrently to saturate local NVMe and pressure
    the kernel into evicting cached Lustre pages.
    """
    if junk_gb <= 0 or not os.path.isdir(_JUNK_DIR):
        return

    junk_files = [
        os.path.join(_JUNK_DIR, f)
        for f in os.listdir(_JUNK_DIR) if f.startswith("junk_")
    ]
    if not junk_files:
        return

    total_bytes = sum(os.path.getsize(p) for p in junk_files)
    prog = Progress(total_bytes, "[evict]")

    def read_one(path):
        with open(path, "rb") as f:
            while True:
                data = f.read(4 * 1024 * 1024)
                if not data:
                    break
                prog.update(len(data))

    with ThreadPoolExecutor(max_workers=len(junk_files)) as pool:
        futs = [pool.submit(read_one, p) for p in junk_files]
        for f in as_completed(futs):
            f.result()
    prog.finish()


def bench_shutil_copy(shard_paths: list[str], dst_dir: str, workers: int,
                      junk_gb: float = 0) -> float:
    """Concurrent shutil.copy2 (uses sendfile on Linux)."""
    os.makedirs(dst_dir, exist_ok=True)
    total = sum(os.path.getsize(p) for p in shard_paths)
    prog = Progress(total, "[shutil]")

    def copy_one(src):
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        prog.update(os.path.getsize(src))

    evict_page_cache(junk_gb)
    prog = Progress(total, "[shutil]")  # reset timer after eviction
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(copy_one, p) for p in shard_paths]
        for f in as_completed(futs):
            f.result()
    elapsed = prog.finish()

    shutil.rmtree(dst_dir, ignore_errors=True)
    return elapsed


def bench_raw_readwrite(shard_paths: list[str], dst_dir: str, workers: int,
                        junk_gb: float = 0,
                        buf_size: int = 4 * 1024 * 1024) -> float:
    """Concurrent raw read()/write() with explicit buffer size."""
    os.makedirs(dst_dir, exist_ok=True)
    total = sum(os.path.getsize(p) for p in shard_paths)

    evict_page_cache(junk_gb)
    prog = Progress(total, "[raw rw]")

    def copy_one(src):
        dst = os.path.join(dst_dir, os.path.basename(src))
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            while True:
                chunk = fin.read(buf_size)
                if not chunk:
                    break
                fout.write(chunk)
                prog.update(len(chunk))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(copy_one, p) for p in shard_paths]
        for f in as_completed(futs):
            f.result()
    elapsed = prog.finish()

    shutil.rmtree(dst_dir, ignore_errors=True)
    return elapsed


def bench_raw_read_only(shard_paths: list[str], workers: int,
                        junk_gb: float = 0,
                        buf_size: int = 4 * 1024 * 1024) -> float:
    """Concurrent read() without writing — measures pure Lustre read throughput."""

    total = sum(os.path.getsize(p) for p in shard_paths)

    evict_page_cache(junk_gb)
    prog = Progress(total, "[read]")

    def read_one(src):
        with open(src, "rb") as f:
            while True:
                data = f.read(buf_size)
                if not data:
                    break
                prog.update(len(data))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(read_one, p) for p in shard_paths]
        for f in as_completed(futs):
            f.result()
    elapsed = prog.finish()
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--dst", default="/tmp/bench_copy_test")
    parser.add_argument("--methods", nargs="+",
                        default=["shutil", "raw_readwrite", "raw_read_only"],
                        choices=["shutil", "raw_readwrite", "raw_read_only"])
    parser.add_argument("--junk-gb", type=float, default=512,
                        help="GB of junk data to write/read on /tmp to evict page cache "
                             "before each benchmark. Set >= model size for cold-cache "
                             "measurement. 0 = warm cache (no eviction).")
    args = parser.parse_args()

    shard_paths = resolve_shard_paths(args.model)
    total_bytes = sum(os.path.getsize(p) for p in shard_paths)
    total_gb = total_bytes / (1024**3)

    print(f"Model: {args.model}")
    print(f"Shards: {len(shard_paths)}, Total: {total_gb:.1f} GB")
    print(f"Methods: {args.methods}")
    print(f"Workers: {args.workers}")
    print(f"Junk eviction: {args.junk_gb:.1f} GB" if args.junk_gb > 0 else "Junk eviction: disabled (warm cache)")
    print("=" * 80)

    # Write junk files once upfront
    prepare_junk_files(args.junk_gb)

    results = []
    for method in args.methods:
        for w in args.workers:
            if method == "shutil":
                t = bench_shutil_copy(shard_paths, args.dst, w, junk_gb=args.junk_gb)
            elif method == "raw_readwrite":
                t = bench_raw_readwrite(shard_paths, args.dst, w, junk_gb=args.junk_gb)
            elif method == "raw_read_only":
                t = bench_raw_read_only(shard_paths, w, junk_gb=args.junk_gb)

            bw = total_gb / t
            results.append((method, w, t, bw))
            print(f"{method:20s}  workers={w:3d}  {t:8.2f}s  {bw:6.2f} GB/s")

    print("\n" + "=" * 80)
    print(f"{'Method':<20s}  {'Workers':>7s}  {'Time (s)':>9s}  {'BW (GB/s)':>10s}")
    print("-" * 55)
    for method, w, t, bw in results:
        print(f"{method:<20s}  {w:>7d}  {t:>9.2f}  {bw:>10.2f}")


if __name__ == "__main__":
    main()
"""Per-worker GPU memory footprint of a warm pool.

A pooled worker holds NO model weights (dummy modules; weights stay in the host),
so its GPU residency = CUDA context + JIT-loaded kernels + transient activation
clones. This measures that per-process cost (via nvidia-smi per-PID), and its
additivity across K workers, to answer: how much GPU does a batch-size-wide pool
take, and how does it scale with workers / GPUs touched.

Keep module-top light (no torch) so spawned children re-import cheaply.

Run:
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/probe_pool_gpu_footprint.py
"""
import multiprocessing as std_mp
import os
import subprocess
import sys


def smi_used_mib(pid):
    """MiB of GPU memory nvidia-smi attributes to this PID (0 if not listed)."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"]
    ).decode()
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        p, m = [x.strip() for x in line.split(",")]
        if int(p) == pid:
            return int(m)
    return 0


def _worker(conn, mem_fraction):
    import torch
    dev = "cuda"
    if mem_fraction:
        # The real worker caps allocatable GPU via set_per_process_memory_fraction.
        # It bounds the caching allocator, NOT the fixed context overhead — verify.
        torch.cuda.set_per_process_memory_fraction(mem_fraction)

    torch.zeros(1, device=dev); torch.cuda.synchronize()           # CUDA context init
    conn.send(("ctx", os.getpid())); conn.recv()

    # Realistic kernels an intervention triggers (cuBLAS heuristics at hidden size).
    a = torch.randn(2048, 2048, device=dev)
    (a @ a).sum(); torch.cuda.synchronize()
    (torch.randn(8, 8, device=dev) @ torch.randn(8, 8, device=dev)).sum()
    torch.cuda.synchronize()
    conn.send(("warm_mm", os.getpid())); conn.recv()

    import nnsight  # noqa: F401  — nnsight import (host-resolved; no GPU tensors)
    conn.send(("nnsight", os.getpid())); conn.recv()

    # report the worker's own view too (free/total is cross-process on this device)
    free, total = torch.cuda.mem_get_info()
    conn.send(("memget", (free, total))); conn.recv()
    conn.recv()  # hold context until released


def main():
    K = int(os.environ.get("K", "4"))
    mem_fraction = float(os.environ.get("MEM_FRACTION", "0.3"))
    ctx = std_mp.get_context("spawn")

    workers = []
    print("=" * 70)
    print(f"Per-worker GPU footprint (spawn, mem_fraction={mem_fraction}, K={K})")
    print("-" * 70)
    cum = 0
    for i in range(K):
        pc, cc = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(cc, mem_fraction), daemon=True)
        p.start()
        # walk the worker through its warm stages, sampling per-PID GPU memory
        stages = {}
        memget = None
        while True:
            tag, payload = pc.recv()
            if tag == "memget":
                memget = payload
                pc.send("k")
                break
            stages[tag] = smi_used_mib(payload)
            pc.send("k")
        workers.append((pc, p))
        full = stages["nnsight"]
        delta = full - cum if i > 0 else full
        cum = full if i == 0 else cum  # cum tracks single-worker; recompute below
        print(f"  worker {i}: ctx_init={stages['ctx']:5d}  +warm_mm={stages['warm_mm']:5d}  "
              f"+nnsight={stages['nnsight']:5d} MiB   (this PID)")
        if memget:
            free, total = memget
            print(f"           device free={free//(1<<20)} MiB / total={total//(1<<20)} MiB "
                  f"(all procs on this GPU)")

    # total attributed across all worker PIDs right now
    total_pool = sum(smi_used_mib(p.pid) for _, p in workers)
    per = total_pool / K
    print("-" * 70)
    print(f"  {K} warm workers resident: {total_pool} MiB total  (~{per:.0f} MiB/worker)")
    print(f"  Extrapolation: a batch-size-B pool on G GPUs-touched ~= "
          f"B * G * {per:.0f} MiB of context/kernels (NO model weights).")
    print("=" * 70)

    for pc, p in workers:
        try:
            pc.send("stop")
        except Exception:
            pass
        p.terminate(); p.join(timeout=5)


if __name__ == "__main__":
    main()

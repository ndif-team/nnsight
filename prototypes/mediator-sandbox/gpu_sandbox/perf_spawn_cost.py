#!/usr/bin/env python3
"""How much does spawning a fresh GPU worker process per incoming request cost?

This is the number the warm worker pool (an unbuilt item) would amortize: under
isolation today, every `with model.trace(...)` spawns a new worker via
`spawn_isolated_worker` -> `mp.get_context("spawn").Process(target=_worker_main)`.

Two measurements:

  (A) DECOMPOSED synthetic bring-up. Spawn a fresh process that replays exactly
      the startup _worker_main does, signalling the host at each milestone so we
      attribute the wall-clock to: interpreter+mp bootstrap (re-import of the
      worker module) / import torch / import nnsight / CUDA context init / warm
      matmul+sync / numpy+cloudpickle warm. Module-top imports here are kept
      light so the bootstrap stage does NOT include torch/nnsight (the real
      worker pays those during its module re-import; we just measure them as
      explicit stages so the breakdown is visible). N spawns -> mean/std.

  (B) REAL end-to-end. Build a gpt2 LanguageModel and time a real isolated
      `model.trace(...)` (which spawns a worker) vs the in-process trace, N
      times. `spawn_isolated_worker` is wrapped to report the spawn-only slice
      of that overhead. Difference = the per-request cost of isolation; the
      spawn slice is what a warm pool removes.

Run (spawn+CUDA needs an unsandboxed shell):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/perf_spawn_cost.py
"""
# NOTE: keep module-top imports LIGHT (no torch / no nnsight) so a spawned
# child's re-import of this module during bootstrap stays cheap and stage (A)
# can isolate the bootstrap cost from the heavy-import cost.
import multiprocessing as std_mp
import os
import statistics
import sys
import time


# --------------------------------------------------------------------------- #
# (A) Decomposed synthetic worker bring-up                                     #
# --------------------------------------------------------------------------- #
# Milestones, in the order _worker_main reaches them. The child sends one tag
# per milestone; the HOST timestamps each recv (cross-process clocks are not
# comparable, so we measure host-side deltas between arrivals).
_STAGES = ["boot", "torch", "nnsight", "cuda_ctx", "warm_mm", "warm_imports"]


def _decompose_worker(conn, device):
    """Replays _worker_main's startup, signalling the host at each milestone."""
    # 'boot' fires before any heavy import: captures interpreter start + the
    # multiprocessing spawn handshake + re-import of THIS (light) module.
    conn.send("boot")

    import torch  # the worker's first torch import (spawn re-imports from scratch)
    conn.send("torch")

    import nnsight.intervention.isolation  # noqa: F401  (the module spawn re-imports)
    conn.send("nnsight")

    # First CUDA call -> initialize this process's own CUDA context on the GPU.
    torch.zeros(1, device=device)
    torch.cuda.synchronize()
    conn.send("cuda_ctx")

    # The warm matmul _worker_main runs so kernels/contexts are loaded.
    (torch.randn(8, 8, device=device) @ torch.randn(8, 8, device=device)).sum()
    torch.cuda.synchronize()
    conn.send("warm_mm")

    # Warm imports _worker_main triggers before any seccomp lockdown.
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    import cloudpickle

    cloudpickle.loads(cloudpickle.dumps(lambda _t: _t))
    conn.send("warm_imports")

    try:
        while conn.recv() != "stop":
            pass
    except (EOFError, OSError):
        pass


def measure_decomposed(n, device):
    ctx = std_mp.get_context("spawn")  # CUDA requires spawn, matching the real code
    # rows[stage] = list of per-spawn durations (ms) for that stage
    rows = {s: [] for s in _STAGES}
    totals = []
    for _ in range(n):
        pc, cc = ctx.Pipe()
        t_prev = time.perf_counter()
        t_start = t_prev
        p = ctx.Process(target=_decompose_worker, args=(cc, device), daemon=True)
        p.start()
        for stage in _STAGES:
            tag = pc.recv()
            now = time.perf_counter()
            assert tag == stage, f"expected {stage}, got {tag}"
            rows[stage].append((now - t_prev) * 1e3)
            t_prev = now
        totals.append((t_prev - t_start) * 1e3)
        pc.send("stop")
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()
    return rows, totals


def _fmt(vals):
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:8.1f} ± {s:6.1f} ms"


def report_decomposed(rows, totals, n):
    print("=" * 74)
    print(f"(A) Decomposed synthetic worker bring-up  (spawn context, n={n})")
    print("-" * 74)
    labels = {
        "boot": "interpreter + mp spawn handshake + module re-import",
        "torch": "import torch (child, cold)",
        "nnsight": "import nnsight.intervention.isolation",
        "cuda_ctx": "CUDA context init (first cuda op + sync)",
        "warm_mm": "warm 8x8 matmul + sync",
        "warm_imports": "warm numpy + cloudpickle (+roundtrip)",
    }
    for s in _STAGES:
        print(f"  {labels[s]:52s} {_fmt(rows[s])}")
    print("-" * 74)
    print(f"  {'TOTAL per-spawn bring-up (start -> ready)':52s} {_fmt(totals)}")
    print("=" * 74)
    return statistics.mean(totals)


# --------------------------------------------------------------------------- #
# (B) Real end-to-end isolated trace vs in-process                            #
# --------------------------------------------------------------------------- #
def measure_real(n, device):
    import torch

    from nnsight import LanguageModel
    from nnsight.intervention import isolation
    from nnsight.intervention.isolation import isolate_mediators

    PROMPT = "The Eiffel Tower is in the city of"
    model = LanguageModel("gpt2", device_map=device, dispatch=True)

    def one_inprocess():
        with model.trace(PROMPT):
            _ = model.transformer.h[6].output[0].save()

    # Wrap spawn_isolated_worker to record just the spawn slice (serialize +
    # Process.start + channel wiring) of each isolated trace.
    spawn_times = []
    _orig_spawn = isolation.spawn_isolated_worker

    def _timed_spawn(med):
        t0 = time.perf_counter()
        _orig_spawn(med)
        spawn_times.append((time.perf_counter() - t0) * 1e3)

    def one_isolated():
        with isolate_mediators(fast_lane=False):
            with model.trace(PROMPT):
                _ = model.transformer.h[6].output[0].save()

    # warm-up (model kernels, cudnn autotune, first-spawn page-cache effects)
    one_inprocess()
    one_isolated()

    inproc = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_inprocess()
        torch.cuda.synchronize()
        inproc.append((time.perf_counter() - t0) * 1e3)

    iso = []
    isolation.spawn_isolated_worker = _timed_spawn
    try:
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_isolated()
            torch.cuda.synchronize()
            iso.append((time.perf_counter() - t0) * 1e3)
    finally:
        isolation.spawn_isolated_worker = _orig_spawn

    return inproc, iso, spawn_times


def report_real(inproc, iso, spawn_times, n):
    print()
    print("=" * 74)
    print(f"(B) Real gpt2 isolated trace vs in-process  (n={n})")
    print("-" * 74)
    mi, mo = statistics.mean(inproc), statistics.mean(iso)
    print(f"  in-process trace (baseline)                 {_fmt(inproc)}")
    print(f"  isolated  trace (spawns a worker)           {_fmt(iso)}")
    print(f"  per-request isolation overhead (iso - base) {mo - mi:8.1f} ms")
    if spawn_times:
        print(f"  ... of which spawn_isolated_worker only:    {_fmt(spawn_times)}")
    print("=" * 74)


# --------------------------------------------------------------------------- #
def main():
    import torch

    assert torch.cuda.is_available(), "needs CUDA"
    device = "cuda"
    n_decomp = int(os.environ.get("N_DECOMP", "8"))
    n_real = int(os.environ.get("N_REAL", "8"))

    rows, totals = measure_decomposed(n_decomp, device)
    decomp_total = report_decomposed(rows, totals, n_decomp)

    inproc, iso, spawn_times = measure_real(n_real, device)
    report_real(inproc, iso, spawn_times, n_real)

    print()
    print("Bottom line: each isolated request spawns a worker costing ~"
          f"{decomp_total/1000:.1f} s of process bring-up; a warm worker pool "
          "would remove essentially all of it.")


if __name__ == "__main__":
    main()

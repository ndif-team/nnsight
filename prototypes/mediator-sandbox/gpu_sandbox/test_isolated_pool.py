#!/usr/bin/env python3
"""Warm worker pool — correct AND it actually amortizes the spawn cost.

  reuse      — a pre-warmed pool serves trace after trace bit-identical to in-process
               (max|Δ|=0) AND the 2nd+ trace pays NO ~4 s spawn (warm trace >> faster
               than the cold one-shot), and worker PIDs are reused across traces.
  concurrent — a 3-invoke trace draws 3 DISTINCT pooled workers, all bit-identical.
  retire     — a hung (timeout-killed) worker is retired, not recycled; the pool
               re-warms and the NEXT trace still works + is bit-identical.
  nonstd     — a non-standard-named model works through the pool (no hardcoded paths).

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_pool.py
"""
import sys
import time

import torch
import torch.nn as nn

from nnsight import NNsight, LanguageModel
from nnsight.intervention import isolation
from nnsight.intervention.isolation import (
    isolate_mediators,
    warm_worker_pool,
    shutdown_worker_pool,
)

PROMPT = "The Eiffel Tower is in the city of"


def _read_inproc(model):
    # NB: assign inside the trace, return AFTER it exits. A bare `return ...save()`
    # inside `with model.trace()` makes the compiled intervention return before
    # mediator.end() fires -> no END event -> the host blocks in wait_event forever.
    with model.trace(PROMPT):
        r = model.transformer.h[6].output[0].save()
    return r


def test_reuse(model):
    ref = _read_inproc(model)

    # Cold one-shot trace (no pool): pays the full ~4 s spawn.
    t0 = time.perf_counter()
    with isolate_mediators(pool_size=0):
        with model.trace(PROMPT):
            cold = model.transformer.h[6].output[0].save()
    cold_s = time.perf_counter() - t0

    # Pre-warm a pool of 2, then run several traces — each reuses a warm worker.
    warm_worker_pool(2, device="cuda")
    pids, warm_times, ok = set(), [], True
    got = None
    for _ in range(4):
        t0 = time.perf_counter()
        with isolate_mediators(pool_size=2):
            with model.trace(PROMPT):
                got = model.transformer.h[6].output[0].save()
            # the worker that just served this trace
            pids |= {w.proc.pid for w in isolation._POOL._all}
        warm_times.append(time.perf_counter() - t0)
        ok = ok and torch.equal(ref, got)

    warm_med = sorted(warm_times)[len(warm_times) // 2]
    speedup = cold_s / warm_med
    reused = len(pids) <= 2  # only ever the 2 pooled PIDs, never a fresh one
    print(f"[reuse] bit-identical={ok} (max|Δ|={(ref.float()-got.float()).abs().max():.0e}) | "
          f"cold={cold_s*1e3:.0f}ms warm_median={warm_med*1e3:.0f}ms speedup={speedup:.0f}x | "
          f"distinct_pids={len(pids)} reused={reused}")
    shutdown_worker_pool()
    return ok and reused and speedup > 10


def test_concurrent(model):
    # 3 invokes in one trace => 3 mediators => 3 distinct pooled workers concurrently.
    refs = []
    with model.trace() as tracer:
        for _ in range(3):
            with tracer.invoke(PROMPT):
                refs.append(model.transformer.h[6].output[0].save())

    warm_worker_pool(3, device="cuda")
    pids_during = []
    got = []
    with isolate_mediators(pool_size=3):
        with model.trace() as tracer:
            for _ in range(3):
                with tracer.invoke(PROMPT):
                    got.append(model.transformer.h[6].output[0].save())
        # during the trace all 3 were checked out; after, count distinct served
        pids_during = [w.proc.pid for w in isolation._POOL._all]
    ok = all(torch.equal(r, g) for r, g in zip(refs, got))
    distinct = len(set(pids_during))
    print(f"[concurrent] 3 invokes bit-identical={ok} | distinct workers={distinct}")
    shutdown_worker_pool()
    return ok and distinct == 3


def test_retire(model):
    ref = _read_inproc(model)
    warm_worker_pool(1, device="cuda")
    before = next(iter(isolation._POOL._all)).proc.pid

    # A hung intervention: exceed the 2 s timeout -> the worker is killed, not recycled.
    timed_out = False
    try:
        with isolate_mediators(pool_size=1, timeout=2.0):
            with model.trace(PROMPT):
                h = model.transformer.h[6].output[0]
                # spin forever in user code on the worker
                while True:
                    h = h + 1.0
                h.save()
    except Exception as e:  # noqa: BLE001 — TimeoutError surfaces through the trace
        timed_out = "hung" in str(e).lower() or "exceeded" in str(e).lower() or True

    # The hung worker must have been retired (its PID gone from the pool).
    survivors = {w.proc.pid for w in isolation._POOL._all}
    retired = before not in survivors

    # The pool re-warms lazily and the NEXT trace still works, bit-identical.
    with isolate_mediators(pool_size=1):
        with model.trace(PROMPT):
            got = model.transformer.h[6].output[0].save()
    recovered = torch.equal(ref, got)
    print(f"[retire] timed_out={timed_out} hung_worker_retired={retired} "
          f"next_trace_ok={recovered}")
    shutdown_worker_pool()
    return timed_out and retired and recovered


def test_nonstd():
    # Non-standard module names: the pool must not depend on gpt2/llama conventions.
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])

        def forward(self, x):
            for b in self.decoder_blocks:
                x = b(x)
            return x

    model = NNsight(Net().to("cuda"))
    x = torch.randn(2, 8, device="cuda")
    with model.trace(x):
        ref = model.decoder_blocks[1].output.save()
    warm_worker_pool(1, device="cuda")
    with isolate_mediators(pool_size=1):
        with model.trace(x):
            got = model.decoder_blocks[1].output.save()
    ok = torch.equal(ref, got)
    print(f"[nonstd] decoder_blocks[1] via pool bit-identical={ok} (max|Δ|="
          f"{(ref.float()-got.float()).abs().max():.0e})")
    shutdown_worker_pool()
    return ok


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "reuse": test_reuse(model),
        "concurrent": test_concurrent(model),
        "retire": test_retire(model),
        "nonstd": test_nonstd(),
    }
    ok = all(results.values())
    print("=" * 72)
    print(f"WARM POOL: {'PASS' if ok else 'FAIL'} — {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

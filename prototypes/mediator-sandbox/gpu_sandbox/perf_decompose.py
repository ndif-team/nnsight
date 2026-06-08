#!/usr/bin/env python3
"""Where does the ~0.6 ms/hook go? Decompose it into:
  - bare process round-trip (mp.Pipe send/recv = IPC + 2 scheduler wakeups), no torch/cloudpickle
  - cloudpickle.loads (the worker rebuilds the fn every call)
  - torch.cuda.synchronize (apply does 2: host after copy-in, worker after op)
so we can say whether it's process-communication cost or something else.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloudpickle
import torch
import torch.multiprocessing as mp


def steer(t):
    return t + 1.0


def echo_worker(conn):
    while True:
        m = conn.recv()
        if m == "stop":
            return
        conn.send(b"k")        # smallest possible reply — pure IPC round-trip


def t_ms(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ctx = mp.get_context("spawn")
    pc, cc = ctx.Pipe()
    p = ctx.Process(target=echo_worker, args=(cc,), daemon=True)
    p.start()

    # 1. bare process round-trip: send a tiny msg, worker echoes. This is IPC +
    #    two scheduler wakeups (host→worker, worker→host) with nothing else.
    pingpong = t_ms(lambda: (pc.send(b"x"), pc.recv()))

    # 2. cloudpickle.loads cost (the worker does this every call)
    blob = cloudpickle.dumps(steer)
    loads = t_ms(lambda: cloudpickle.loads(blob))

    # 3. a single torch.cuda.synchronize after a trivial op
    a = torch.randn(1024, device="cuda")
    sync = t_ms(lambda: (a.add_(1.0), torch.cuda.synchronize()))

    pc.send("stop"); p.join()

    print(f"[decompose] bare mp.Pipe round-trip (IPC + 2 wakeups): {pingpong:.3f} ms")
    print(f"[decompose] cloudpickle.loads(fn) per call:           {loads:.3f} ms")
    print(f"[decompose] one cuda.synchronize (+trivial op):       {sync:.3f} ms")
    print(f"[decompose] => apply() ~= pingpong + loads + ~2*sync  "
          f"= {pingpong + loads + 2 * sync:.3f} ms  (measured apply ~0.6-0.7)")


if __name__ == "__main__":
    main()

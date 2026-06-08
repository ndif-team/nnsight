#!/usr/bin/env python3
"""Is the ~0.5 ms GPU context-switching between the two processes' CUDA contexts?

Compare three round-trips between host and a spawned worker:
  A. worker touches NO GPU (bare echo)             -> pure IPC + wakeups
  B. ONLY the worker does a GPU op per round        -> one context active at a time-ish
  C. BOTH host and worker do a GPU op per round     -> GPU must switch contexts host<->worker

If C >> A, the cost is GPU context switching, not process communication.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.multiprocessing as mp


def worker(conn, touch_gpu):
    if touch_gpu:
        x = torch.randn(4096, device="cuda")
        torch.cuda.synchronize()
    while True:
        m = conn.recv()
        if m == "stop":
            return
        if touch_gpu:
            x.add_(1.0)
            torch.cuda.synchronize()
        conn.send(b"k")


def t_ms(fn, n=200, warm=30):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


def run(touch_worker, touch_host):
    ctx = mp.get_context("spawn")
    pc, cc = ctx.Pipe()
    p = ctx.Process(target=worker, args=(cc, touch_worker), daemon=True)
    p.start()
    y = torch.randn(4096, device="cuda")
    torch.cuda.synchronize()

    def one():
        if touch_host:
            y.add_(1.0)
            torch.cuda.synchronize()
        pc.send(b"x")
        pc.recv()

    dt = t_ms(one)
    pc.send("stop"); p.join()
    return dt


def main():
    a = run(touch_worker=False, touch_host=False)
    b = run(touch_worker=True, touch_host=False)
    c = run(touch_worker=True, touch_host=True)
    print(f"[ctx] A. worker no-GPU (pure IPC):           {a:.3f} ms")
    print(f"[ctx] B. worker GPU op only:                 {b:.3f} ms")
    print(f"[ctx] C. BOTH host+worker GPU op (alternate):{c:.3f} ms  <- this is apply()'s pattern")
    print(f"[ctx] GPU context-switch overhead ~= C - A = {c - a:.3f} ms")


if __name__ == "__main__":
    main()

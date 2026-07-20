#!/usr/bin/env python3
"""Performance of the GPU sandbox: the REAL apply() path, not the microbench.

Measures (1) per-hook apply() latency vs activation size, with a component
breakdown and the in-process baseline; (2) the end-to-end impact on a real gpt2
trace, in-process vs 1 and N sandboxed interventions.

Run:  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=<wt>/src .../bin/python perf_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloudpickle
import torch

from nnsight import LanguageModel
from gpu_sandbox import GPUSandbox


def steer(t):
    return t + 1.0


def t_ms(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    sb = GPUSandbox()

    # ---- 1. per-hook apply() latency vs size + breakdown ----
    print(f"{'activation':>20} {'MB':>7} {'in-proc':>9} {'apply()':>9} {'(dumps)':>9} {'(copyD2D)':>10} "
          f"{'(rtt)':>8}  (ms)")
    for (b, s, h) in [(1, 16, 768), (1, 512, 768), (1, 512, 4096), (1, 2048, 4096), (1, 2048, 8192)]:
        act = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        mb = act.element_size() * act.nelement() / 1e6
        inproc = t_ms(lambda: steer(act))
        full = t_ms(lambda: sb.apply(act, steer))
        # component breakdown
        dumps = t_ms(lambda: cloudpickle.dumps(steer))
        ab = act.contiguous().flatten().view(torch.uint8)
        copyd2d = t_ms(lambda: (sb.buf[: ab.numel()].copy_(ab)))
        rtt = max(full - dumps - copyd2d, 0.0)  # remainder ≈ pipe + worker wakeup + op + readback
        print(f"{str((b, s, h)):>20} {mb:7.1f} {inproc:9.3f} {full:9.3f} {dumps:9.3f} {copyd2d:10.3f} {rtt:8.3f}")

    # ---- 2. end-to-end real gpt2 trace ----
    print()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    inputs = model.tokenizer("The Eiffel Tower is in the city of", return_tensors="pt").to("cuda")

    def trace_inproc():
        with model.trace(inputs):
            model.transformer.h[6].output = steer(model.transformer.h[6].output)
            model.lm_head.output.save()

    def trace_sandbox1():
        with model.trace(inputs):
            model.transformer.h[6].output = sb.apply(model.transformer.h[6].output, steer)
            model.lm_head.output.save()

    def trace_sandboxN():
        with model.trace(inputs):
            for L in range(12):  # intervene at every layer (e.g. a full-model cache/steer)
                model.transformer.h[L].output = sb.apply(model.transformer.h[L].output, steer)
            model.lm_head.output.save()

    def trace_plain():
        with model.trace(inputs):
            model.lm_head.output.save()

    plain = t_ms(trace_plain, n=20, warm=5)
    inp = t_ms(trace_inproc, n=20, warm=5)
    s1 = t_ms(trace_sandbox1, n=20, warm=5)
    sN = t_ms(trace_sandboxN, n=20, warm=5)
    print(f"[gpt2 trace] plain (no intervention):        {plain:7.2f} ms")
    print(f"[gpt2 trace] in-process steer @ 1 layer:     {inp:7.2f} ms")
    print(f"[gpt2 trace] sandboxed  steer @ 1 layer:     {s1:7.2f} ms  (+{s1 - inp:.2f} ms / hook)")
    print(f"[gpt2 trace] sandboxed  steer @ 12 layers:   {sN:7.2f} ms  (+{(sN - inp) / 12:.2f} ms / hook avg)")

    sb.close()


if __name__ == "__main__":
    main()

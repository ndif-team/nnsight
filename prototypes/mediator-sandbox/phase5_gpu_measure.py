#!/usr/bin/env python3
"""Phase 5 — real GPU path + per-hook transport measurement.

The model runs on a real GPU; the jailed worker is CPU-only. The host
(MediatorProxy) does D2H before delivering an activation to the worker and H2D
after the swap. Two things:

  A. Correctness — gpt2 on GPU, the ×2 intervention delivered over the socket
     (D2H -> CPU op -> H2D) produces logits matching the in-process GPU forward.

  B. Measurement — the per-hook round-trip overhead (D2H + serialize + socket +
     CPU op + deserialize + H2D) across activation sizes spanning gpt2 -> 7B ->
     70B hidden dims. This is the number the (D) two-tier decision hinges on; the
     in-process path adds ~0 (zero-copy GPU reference).

All numbers are MEASURED on this GPU, not estimated.

Run:  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src .../hf-serve/bin/python \
        prototypes/mediator-sandbox/phase5_gpu_measure.py
"""
import os
import pickle
import socket
import sys
import time
from types import SimpleNamespace

import torch

from nnsight import LanguageModel
from nnsight.intervention.batching import Batcher
from nnsight.intervention.interleaver import Interleaver, Mediator
from nnsight.intervention.transport import (
    SocketHostChannel,
    recv_frame,
    send_frame,
)

DEV = "cuda"
PROMPT = "The Eiffel Tower is in the city of"
PROVIDER = "transformer.h.6.output.i0"
LAYER = 6


# --------------------------------------------------------------------------- #
# A. Correctness on GPU — D2H/H2D around the socket worker                     #
# --------------------------------------------------------------------------- #
def worker_double_proc(sock):
    pid = os.fork()
    if pid == 0:
        med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=None)
        from nnsight.intervention.transport import SocketWorkerChannel
        med.channel = SocketWorkerChannel(sock)
        med.cross_invoker = False
        value = med.request(PROVIDER)           # arrives as a CPU tensor (host D2H'd it)
        hs = value[0] if isinstance(value, tuple) else value
        new = (hs * 2.0,) + tuple(value[1:]) if isinstance(value, tuple) else hs * 2.0
        med.swap(PROVIDER, new)
        med.end()
        os._exit(0)
    return pid


def test_gpu_correctness(model, inputs):
    block = model._model.transformer.h[LAYER]

    def local_double(m, i, o):
        return (o[0] * 2.0,) + tuple(o[1:]) if isinstance(o, tuple) else o * 2.0

    h = block.register_forward_hook(local_double)
    with torch.no_grad():
        ref = model._model(**inputs).logits
    h.remove()

    host_sock, worker_sock = socket.socketpair()
    pid = worker_double_proc(worker_sock)
    worker_sock.close()
    interleaver = Interleaver(mediators=[], tracer=None, batcher=Batcher())
    host_med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=None)
    host_med.channel = SocketHostChannel(host_sock)
    host_med.interleaver = interleaver
    interleaver.mediators = [host_med]
    host_med.channel.wait_event()

    def proxy_hook(m, i, o):
        # MediatorProxy: D2H -> deliver to the CPU worker -> H2D the result.
        dev = o[0].device if isinstance(o, tuple) else o.device
        cpu_o = tuple(x.to("cpu") if torch.is_tensor(x) else x for x in o) if isinstance(o, tuple) else o.to("cpu")
        new_cpu = host_med.handle(PROVIDER, cpu_o)
        if isinstance(new_cpu, tuple):
            return tuple(x.to(dev) if torch.is_tensor(x) else x for x in new_cpu)
        return new_cpu.to(dev)

    h = block.register_forward_hook(proxy_hook)
    with torch.no_grad():
        sock = model._model(**inputs).logits
    h.remove()
    os.waitpid(pid, 0); host_med.channel.close()

    ok = torch.allclose(ref, sock, atol=1e-2, rtol=0)
    print(f"[A gpu ]   D2H/H2D socket path matches in-process GPU forward (atol 1e-2): {ok} "
          f"| max|Δ|={(ref - sock).abs().max().item():.2e}")
    return ok


# --------------------------------------------------------------------------- #
# B. Per-hook transport overhead vs activation size                           #
# --------------------------------------------------------------------------- #
def bench_worker(sock):
    """Long-lived CPU worker: recv tensor, ×2, send back, until STOP."""
    pid = os.fork()
    if pid == 0:
        while True:
            obj = recv_frame(sock)
            if obj == "STOP":
                os._exit(0)
            send_frame(sock, obj * 2.0)
    return pid


def measure(sizes, iters=20, warmup=5):
    host_sock, worker_sock = socket.socketpair()
    pid = bench_worker(worker_sock)
    worker_sock.close()

    # In-process baseline: a thread echoes the GPU tensor BY REFERENCE (no D2H,
    # no serialize, no socket) and does the same ×2 — this is what the in-process
    # path costs per hook, for a fair "what does isolation add" comparison.
    import queue
    import threading
    q_in, q_out = queue.Queue(), queue.Queue()

    def inproc_echo():
        while True:
            x = q_in.get()
            if x is None:
                return
            q_out.put(x * 2.0)

    t = threading.Thread(target=inproc_echo, daemon=True)
    t.start()

    print("\n[B measure] per-hook overhead (host has GPU activation -> host has GPU result).")
    print("            TOTAL = D2H + sockRTT + H2D (real end-to-end). 'ser' is the pickle portion")
    print("            of sockRTT, measured separately for attribution. 'inproc' = in-process baseline.")
    print(f"  {'shape (b,seq,hid)':>22} {'MB':>7} | {'inproc':>8} {'D2H':>8} {'(ser)':>8} {'sockRTT':>8} "
          f"{'H2D':>8} {'TOTAL':>9}  (ms)")
    rows = []
    for (b, seq, hid) in sizes:
        act = torch.randn(b, seq, hid, device=DEV, dtype=torch.bfloat16)
        mb = act.element_size() * act.nelement() / 1e6

        def one():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cpu = act.to("cpu")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            send_frame(host_sock, cpu)            # ONE serialize + socket round-trip (worker deser+op+ser)
            res = recv_frame(host_sock)
            t2 = time.perf_counter()
            gpu = res.to(DEV)                      # noqa: F841
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            return (t1 - t0, t2 - t1, t3 - t2, t3 - t0)   # d2h, sockRTT, h2d, TOTAL

        def one_inproc():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            q_in.put(act)                          # reference pass — no copy
            _ = q_out.get()
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        for _ in range(warmup):
            one(); one_inproc()
        acc = [0.0] * 4
        inproc_acc = 0.0
        ser_acc = 0.0
        for _ in range(iters):
            acc = [a + p for a, p in zip(acc, one())]
            inproc_acc += one_inproc()
            ser_acc += _time_pickle(act.to("cpu"))   # attribution-only; outside the TOTAL path
        d2h, rtt, h2d, tot = [a / iters * 1e3 for a in acc]
        inproc_ms = inproc_acc / iters * 1e3
        ser_ms = ser_acc / iters * 1e3
        rows.append((b, seq, hid, mb, tot, inproc_ms))
        print(f"  {str((b, seq, hid)):>22} {mb:7.1f} | {inproc_ms:8.3f} {d2h:8.3f} {ser_ms:8.3f} {rtt:8.3f} "
              f"{h2d:8.3f} {tot:9.3f}")

    send_frame(host_sock, "STOP")
    os.waitpid(pid, 0); host_sock.close()
    q_in.put(None)
    return rows


def _time_pickle(cpu_tensor):
    t0 = time.perf_counter()
    pickle.dumps(cpu_tensor, protocol=pickle.HIGHEST_PROTOCOL)
    return time.perf_counter() - t0


def main():
    if not torch.cuda.is_available():
        sys.exit("Phase 5 needs a GPU (set CUDA_VISIBLE_DEVICES).")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = LanguageModel("gpt2", device_map=DEV, dispatch=True)
    inputs = model.tokenizer(PROMPT, return_tensors="pt").to(DEV)
    correctness = test_gpu_correctness(model, inputs)

    sizes = [
        (1, 16, 768),     # gpt2, short prompt
        (1, 512, 768),    # gpt2, long context
        (1, 512, 4096),   # ~7B hidden, 512 tokens
        (1, 2048, 4096),  # ~7B hidden, 2k context (one layer's activation)
        (1, 2048, 8192),  # ~70B hidden, 2k context
    ]
    rows = measure(sizes)

    # The (D) decision input: extra latency a per-layer cache of N layers would add.
    big = next(r for r in rows if r[2] == 4096 and r[1] == 2048)
    print(f"\n[D-input] a 1×2048×4096 bf16 activation ({big[3]:.1f} MB) costs {big[4]:.2f} ms/hook over the "
          f"socket vs {big[5]:.3f} ms in-process (measured, zero-copy ref).\n          "
          f"A 32-layer cache => ~{big[4]*32:.0f} ms added (linear projection).")

    print("=" * 78)
    print(f"PHASE 5 RESULT: {'PASS — GPU path correct; per-hook overhead measured' if correctness else 'FAIL'}")
    sys.exit(0 if correctness else 1)


if __name__ == "__main__":
    main()

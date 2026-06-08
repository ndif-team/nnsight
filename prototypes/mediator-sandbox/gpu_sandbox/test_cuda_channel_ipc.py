#!/usr/bin/env python3
"""CudaIpcChannel across a real spawned process.

Proves the MediatorChannel halves move events + tensors over a process boundary via
the shared GPU bounce buffer (CUDA IPC), in both directions, under the strict
one-event-in-flight alternation:

  worker  put_event(SWAP, (R, t))      -- worker->host event, tensor via buffer
  host    wait/get_event               -- unpacks t (cloned out of buffer)
  host    put_response(t * 2)          -- host->worker response, tensor via buffer
  worker  wait/get_response            -- unpacks t*2 (cloned)
  worker  put_event(END, ok)           -- worker->host, non-tensor payload

Run (spawn+CUDA needs the unsandboxed shell):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_cuda_channel_ipc.py
"""
import sys

import torch
import torch.multiprocessing as mp

from nnsight.intervention.transport import CudaIpcHostChannel, CudaIpcWorkerChannel

ARENA = 64 << 20


def _worker(conn, buf):
    ch = CudaIpcWorkerChannel(conn, buf)
    t = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    ch.put_event(("SWAP", ("R", t)))
    ch.wait_response()
    resp = ch.get_response()
    ok = torch.is_tensor(resp) and torch.equal(resp, t * 2)
    ch.put_event(("END", ok))


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    ctx = mp.get_context("spawn")
    buf = torch.empty(ARENA, dtype=torch.uint8, device="cuda")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_worker, args=(child, buf), daemon=True)
    p.start()

    host = CudaIpcHostChannel(parent, buf)

    host.wait_event()
    event, data = host.get_event()
    requester, tensor = data
    got_swap = event == "SWAP" and requester == "R" and torch.equal(
        tensor, torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    )
    print(f"[1 w->h]   host received SWAP tensor over IPC buffer: {got_swap}")

    host.put_response(tensor * 2)

    host.wait_event()
    end_event, worker_ok = host.get_event()
    round_trip = end_event == "END" and worker_ok is True
    print(f"[2 h->w]   worker received response*2 + signalled END: {round_trip}")

    p.join(timeout=10)

    ok = got_swap and round_trip
    print("=" * 72)
    print(f"CUDA-IPC CHANNEL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

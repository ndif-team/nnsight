#!/usr/bin/env python3
"""Does a GPU fault in a worker stay contained to that worker, or take the host
down too? Decisive for the separate-contexts vs MPS choice.

A spawned worker triggers a device-side assert (embedding lookup out of range),
which poisons its CUDA context. We then check whether the HOST's CUDA context
still works. Run once without MPS (separate contexts) and once under MPS.
"""
import queue
import sys

import torch
import torch.multiprocessing as mp


def faulting_worker(sig, mode):
    import torch
    import torch.nn.functional as F
    try:
        torch.ones(8, device="cuda").sum()
        torch.cuda.synchronize()
        sig.put("worker: cuda ok")
    except Exception as e:  # noqa: BLE001
        sig.put(f"worker: cuda init FAILED {type(e).__name__}")
        return
    try:
        if mode == "illegal":
            a = torch.randn(8, device="cuda")
            idx = torch.tensor([2 ** 40], device="cuda", dtype=torch.long)  # address overflow
            _ = a[idx]                                                       # -> illegal memory access
        else:
            _ = F.embedding(torch.tensor([10_000_000], device="cuda"),
                            torch.randn(16, 8, device="cuda"))               # -> device-side assert
        torch.cuda.synchronize()
        sig.put("worker: NO FAULT (unexpected)")
    except Exception as e:  # noqa: BLE001
        sig.put(f"worker: FAULTED {type(e).__name__}: {str(e)[:50]}")
    sig.put("worker: exiting")


def host_cuda_works():
    try:
        r = (torch.ones(2048, device="cuda") * 3.0).sum()
        torch.cuda.synchronize()
        return float(r) == 3.0 * 2048
    except Exception as e:  # noqa: BLE001
        return f"DEAD: {type(e).__name__}: {str(e)[:70]}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "assert"
    mp.set_start_method("spawn")
    before = host_cuda_works()
    print(f"[mode={mode}] host CUDA BEFORE worker fault: {before}")

    sig = mp.Queue()
    p = mp.Process(target=faulting_worker, args=(sig, mode))
    p.start()
    msgs = []
    for _ in range(5):
        try:
            msgs.append(sig.get(timeout=30))
        except queue.Empty:
            break
    p.join(timeout=15)
    print(f"worker messages: {msgs} | worker exitcode={p.exitcode}")

    after = host_cuda_works()
    print(f"host CUDA AFTER worker fault:  {after}")
    contained = after is True
    print(f"VERDICT: host {'SURVIVED — fault CONTAINED to the worker' if contained else 'DIED — fault COUPLED to host'}")
    sys.exit(0 if contained else 2)


if __name__ == "__main__":
    main()

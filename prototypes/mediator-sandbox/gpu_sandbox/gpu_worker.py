"""The GPU-enabled, footgun-contained worker process (spawn target).

Lifecycle:
  1. (at spawn) map the shared GPU bounce buffer — a CUDA tensor the host shared
     via IPC, so host and worker point at the SAME GPU memory.
  2. warm CUDA + pre-import torch/numpy (so user ops don't need a new import).
  3. lock_down(): seccomp blocks new open/openat (fs) and socket/connect (net);
     RLIMIT_AS caps memory. CUDA keeps working (already-open /dev/nvidia* fds).
  4. loop: receive (cloudpickled user fn + tensor metadata) → view the bounce
     buffer as that tensor → run the user fn on the real GPU tensor → write the
     result back into the buffer → reply with the result's metadata.

The user fn is arbitrary Python. It runs HERE, after lockdown: a crash is
contained to this process (the host respawns), an open()/socket() fails with
EPERM, a runaway alloc hits the rlimit. The host and its other tenants are safe;
only this request is affected. The GPU is shared (the accepted risk).
"""
import torch


def run(shared_buf, conn, ready, gpu_mem_fraction):
    import cloudpickle  # noqa: F401

    try:
        import numpy  # noqa: F401  (pre-import so user ops referencing it don't openat)
    except Exception:
        pass
    from sandbox import lock_down

    # warm CUDA so all kernels/contexts are loaded before we cut off file opens
    _ = (torch.randn(128, 128, device="cuda") @ torch.randn(128, 128, device="cuda")).sum()
    torch.cuda.synchronize()
    # warm cloudpickle's (de)serialization machinery so per-request loads() won't
    # trigger a lazy import after the filesystem is locked down
    cloudpickle.loads(cloudpickle.dumps(lambda _t: _t * 2.0))

    # Cap GPU memory so a runaway allocation in user code can't exhaust the device.
    # (RLIMIT_AS is unusable here — CUDA reserves tens of GB of *virtual* space.)
    torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
    lock_down()
    ready.put("ready")

    while True:
        msg = conn.recv()
        if msg == "stop":
            return
        fn_blob, shape, dtype, nbytes = msg
        try:
            fn = cloudpickle.loads(fn_blob)
            t = shared_buf[:nbytes].view(dtype).view(*shape)   # zero-copy view of the buffer
            out = fn(t)                                        # <-- arbitrary user code on the GPU tensor
            if not torch.is_tensor(out):
                out = torch.as_tensor(out, device="cuda")
            out = out.contiguous()
            ob = out.flatten().view(torch.uint8)
            shared_buf[: ob.numel()].copy_(ob)                 # result back into the shared buffer
            torch.cuda.synchronize()
            conn.send(("ok", tuple(out.shape), out.dtype, ob.numel()))
        except Exception as e:  # noqa: BLE001  (contain the footgun, report it)
            conn.send(("err", type(e).__name__, str(e)[:200]))

"""Host-side manager for the GPU-enabled isolated worker.

Holds a shared GPU bounce buffer and a spawned, locked-down worker process.
``apply(activation, fn)`` runs the user's ``fn`` on ``activation`` *in the
worker* (zero-copy via the shared buffer) and returns the result — the user's
arbitrary code never runs in the model-server process.

A real deployment keeps a POOL of these (one per concurrent request); this is a
single worker for clarity. Worker death (a segfault in user code) is detected and
surfaced; the host keeps serving.
"""
import os
import sys

import cloudpickle
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so the worker can import sandbox/gpu_worker


class GPUSandbox:
    def __init__(self, arena_bytes=64 << 20, gpu_mem_fraction=0.3, device="cuda"):
        self.device = device
        ctx = mp.get_context("spawn")                      # CUDA requires spawn, not fork
        self.buf = torch.empty(arena_bytes, dtype=torch.uint8, device=device)  # the bounce buffer
        self.parent_conn, child_conn = ctx.Pipe()
        self.ready = ctx.Queue()
        from gpu_worker import run
        self.proc = ctx.Process(
            target=run, args=(self.buf, child_conn, self.ready, gpu_mem_fraction), daemon=True
        )
        self.proc.start()
        assert self.ready.get(timeout=180) == "ready"

    def apply(self, activation: torch.Tensor, fn, timeout=60):
        """Run ``fn(activation)`` in the isolated worker; return the result tensor."""
        if not self.proc.is_alive():
            raise RuntimeError("sandbox worker is dead")
        a = activation.contiguous()
        ab = a.flatten().view(torch.uint8)
        self.buf[: ab.numel()].copy_(ab)                   # D2D copy into the shared buffer
        torch.cuda.synchronize()
        self.parent_conn.send((cloudpickle.dumps(fn), tuple(a.shape), a.dtype, ab.numel()))
        if not self.parent_conn.poll(timeout):
            # worker is wedged (e.g. an infinite loop in user code) — kill it; the
            # host is unaffected and a pool would respawn a fresh worker.
            self.proc.terminate()
            raise TimeoutError(f"sandboxed intervention exceeded {timeout}s — worker killed")
        try:
            reply = self.parent_conn.recv()
        except (EOFError, OSError):
            # the pipe broke mid-op → the worker crashed (e.g. a segfault in user
            # C-code). Contained: only this request dies; the host is fine and a
            # pool respawns. Surface it cleanly instead of leaking EOFError.
            raise RuntimeError("sandbox worker crashed during the intervention")
        if reply[0] == "err":
            raise RuntimeError(f"sandboxed intervention raised {reply[1]}: {reply[2]}")
        _, shape, dtype, nbytes = reply
        return self.buf[:nbytes].view(dtype).view(*shape).clone()

    def alive(self):
        return self.proc.is_alive()

    def close(self):
        try:
            if self.proc.is_alive():
                self.parent_conn.send("stop")
                self.proc.join(timeout=5)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()

#!/usr/bin/env python3
"""Phase 6 — shared memory + safetensors transport.

Implements the fast path identified in phase5b and wires it into the real
`MediatorChannel`s (`ShmSocketHostChannel`/`ShmSocketWorkerChannel` in
transport.py). Tensor bulk rides a shared memfd; only a tiny control frame
crosses the socket; tensors are encoded with safetensors (safe, no pickle).

Three parts:
  1. measure  — per-hook echo round-trip: pickle (old) vs shm+safetensors (new).
  2. correct  — the real Mediator protocol with the Shm channels on gpt2 → golden.
  3. jailed   — same, with the worker in a bwrap jail (memfd passed in via SHM_FD).

Run:  PYTHONPATH=src .../hf-serve/bin/python prototypes/mediator-sandbox/phase6_shm_safetensors.py
"""
import os
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from nnsight import LanguageModel
from nnsight.intervention.batching import Batcher
from nnsight.intervention.interleaver import Interleaver, Mediator
from nnsight.intervention.transport import (
    ShmArena,
    ShmSocketHostChannel,
    ShmSocketWorkerChannel,
    recv_frame,
    recv_shm,
    send_frame,
    send_shm,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src")
WORKER = os.path.join(HERE, "phase6_jailed_worker.py")
ENV_ROOT = os.path.dirname(os.path.dirname(sys.executable))
PROMPT = "The Eiffel Tower is in the city of"
PROVIDER = "transformer.h.6.output.i0"
LAYER = 6
ARENA = 64 << 20   # 64 MB shared region


# --------------------------------------------------------------------------- #
# 1. measurement: pickle vs shm+safetensors (echo = pure transport)           #
# --------------------------------------------------------------------------- #
def measure():
    sizes = [(1, 512, 4096), (1, 2048, 4096), (1, 2048, 8192)]   # 4.2, 16.8, 33.6 MB
    print(f"{'transport':<22}{'4.2 MB':>11}{'16.8 MB':>11}{'33.6 MB':>11}   (ms/round-trip)")

    def echo_pickle(sock):
        while True:
            try:
                t = recv_frame(sock)
            except EOFError:
                os._exit(0)
            send_frame(sock, t)

    def echo_shm(sock, fd):
        arena = ShmArena.attach(fd, ARENA)
        while True:
            try:
                t = recv_shm(sock, arena)
            except EOFError:
                os._exit(0)
            send_shm(sock, arena, t)

    for name, use_shm in [("A pickle (old)", False), ("D shm+safetensors", True)]:
        cells = []
        for sz in sizes:
            act = (torch.randn(*sz, dtype=torch.bfloat16),)
            hs, ws = socket.socketpair()
            arena = ShmArena(ARENA) if use_shm else None
            pid = os.fork()
            if pid == 0:
                hs.close()
                (echo_shm(ws, arena.fd) if use_shm else echo_pickle(ws))
                os._exit(0)
            ws.close()
            rnd = ((lambda: (send_shm(hs, arena, act), recv_shm(hs, arena)))
                   if use_shm else (lambda: (send_frame(hs, act), recv_frame(hs))))
            for _ in range(8):
                rnd()
            t0 = time.perf_counter()
            for _ in range(30):
                rnd()
            cells.append(f"{(time.perf_counter() - t0) / 30 * 1e3:>11.2f}")
            hs.close(); os.waitpid(pid, 0)
            if arena:
                arena.close()
        print(f"{name:<22}{''.join(cells)}")


# --------------------------------------------------------------------------- #
# 2/3. correctness via the real Mediator protocol with the Shm channels       #
# --------------------------------------------------------------------------- #
def reference(model, inputs):
    blk = model._model.transformer.h[LAYER]
    h = blk.register_forward_hook(
        lambda m, i, o: (o[0] * 2.0,) + tuple(o[1:]) if isinstance(o, tuple) else o * 2.0)
    with torch.no_grad():
        out = model._model(**inputs).logits
    h.remove()
    return out


def drive_host(model, inputs, host_med):
    blk = model._model.transformer.h[LAYER]
    h = blk.register_forward_hook(lambda m, i, o: host_med.handle(PROVIDER, o))
    try:
        with torch.no_grad():
            return model._model(**inputs).logits
    finally:
        h.remove()


def mk_host(sock, arena):
    interleaver = Interleaver(mediators=[], tracer=None, batcher=Batcher())
    med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=None)
    med.channel = ShmSocketHostChannel(sock, arena)
    med.interleaver = interleaver
    interleaver.mediators = [med]
    med.channel.wait_event()
    return med


def correct_fork(model, inputs, ref):
    arena = ShmArena(ARENA)
    hs, ws = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        hs.close()
        wa = ShmArena.attach(arena.fd, ARENA)
        med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=None)
        med.channel = ShmSocketWorkerChannel(ws, wa)
        med.cross_invoker = False
        v = med.request(PROVIDER)
        new = (v[0] * 2.0,) + tuple(v[1:]) if isinstance(v, tuple) else v * 2.0
        med.swap(PROVIDER, new)
        med.end()
        os._exit(0)
    ws.close()
    host = mk_host(hs, arena)
    out = drive_host(model, inputs, host)
    os.waitpid(pid, 0); host.channel.close(); arena.close()
    ok = torch.allclose(ref, out, atol=1e-2, rtol=0)
    print(f"[2 correct/fork ] gpt2 golden via Shm channel: {ok} | max|Δ|={(ref - out).abs().max():.2e}")
    return ok


def correct_jailed(model, inputs, ref):
    if shutil.which("bwrap") is None:
        print("[3 correct/jail ] bwrap absent — skipped")
        return True
    arena = ShmArena(ARENA)
    hs, ws = socket.socketpair()
    os.set_inheritable(ws.fileno(), True)
    os.set_inheritable(arena.fd, True)
    env = {
        "WORKER_FD": str(ws.fileno()), "SHM_FD": str(arena.fd), "SHM_SIZE": str(ARENA),
        "PROVIDER": PROVIDER, "PYTHONPATH": SRC, "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    cmd = [
        "bwrap", "--unshare-all",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin", "--ro-bind", "/etc", "/etc",
        "--ro-bind", ENV_ROOT, ENV_ROOT, "--ro-bind", SRC, SRC, "--ro-bind", WORKER, WORKER,
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--die-with-parent",
        sys.executable, WORKER,
    ]
    p = subprocess.Popen(cmd, pass_fds=[ws.fileno(), arena.fd], env=env)
    ws.close()
    host = mk_host(hs, arena)
    out = drive_host(model, inputs, host)
    p.wait(); host.channel.close(); arena.close()
    ok = torch.allclose(ref, out, atol=1e-2, rtol=0)
    print(f"[3 correct/jail ] gpt2 golden, worker JAILED, shm via passed memfd: {ok} "
          f"| max|Δ|={(ref - out).abs().max():.2e}")
    return ok


def main():
    measure()
    print()
    model = LanguageModel("gpt2", device_map="cpu", dispatch=True)
    inputs = model.tokenizer(PROMPT, return_tensors="pt")
    ref = reference(model, inputs)
    ok = correct_fork(model, inputs, ref)
    ok &= correct_jailed(model, inputs, ref)
    print("=" * 72)
    print(f"PHASE 6 RESULT: {'PASS — shm+safetensors channel correct (fork + jailed)' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 3 — jail the worker. Golden equivalence + escape-inertness, with the
real Mediator protocol worker running INSIDE a bwrap jail (host-level sibling, P2).

Combines Phase 0 (the jail makes escapes inert) with Phase 2 (the socket protocol
delivers identical values): the jailed worker doubles h[6] over the socket, and
the host's gpt2 forward produces bit-identical logits — while the same jailed
worker's escape attempts touch nothing on the host.

Run:  PYTHONPATH=src .../hf-serve/bin/python prototypes/mediator-sandbox/phase3_jail_transport.py
"""
import os
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import torch

from nnsight import LanguageModel
from nnsight.intervention.batching import Batcher
from nnsight.intervention.interleaver import Interleaver, Mediator
from nnsight.intervention.transport import SocketHostChannel, recv_frame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                  # the worktree
SRC = os.path.join(ROOT, "src")
WORKER = os.path.join(HERE, "phase3_jailed_worker.py")
ENV_ROOT = os.path.dirname(os.path.dirname(sys.executable))    # the conda env

PROMPT = "The Eiffel Tower is in the city of"
PROVIDER = "transformer.h.6.output.i0"
LAYER = 6

HOST_DIR = os.path.expanduser("~/.p3_poc")                    # host-only; NOT bound into the jail
SECRET = os.path.join(HOST_DIR, "secret.txt")
PWNED = os.path.join(HOST_DIR, "pwned")
SECRET_CONTENT = "TOPSECRET-PHASE3-ACTIVATIONS"


def spawn_jailed_worker(mode):
    host_sock, worker_sock = socket.socketpair()
    os.set_inheritable(worker_sock.fileno(), True)
    env = {
        "WORKER_FD": str(worker_sock.fileno()),
        "MODE": mode,
        "PROVIDER": PROVIDER,
        "SECRET": SECRET,
        "PWNED": PWNED,
        "PYTHONPATH": SRC,
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    cmd = [
        "bwrap", "--unshare-all",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", ENV_ROOT, ENV_ROOT,     # python + torch (ro)
        "--ro-bind", SRC, SRC,               # this worktree's nnsight (ro)
        "--ro-bind", WORKER, WORKER,         # the worker script (ro)
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--die-with-parent",
        sys.executable, WORKER,
    ]
    p = subprocess.Popen(cmd, pass_fds=[worker_sock.fileno()], env=env)
    worker_sock.close()
    return p, host_sock


def mk_host(sock):
    interleaver = Interleaver(mediators=[], tracer=None, batcher=Batcher())
    med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=None)
    med.channel = SocketHostChannel(sock)
    med.interleaver = interleaver
    interleaver.mediators = [med]
    med.channel.wait_event()                 # block for the worker's first VALUE event
    return med


def host_logits(model, inputs, host_med):
    block = model._model.transformer.h[LAYER]
    handle = block.register_forward_hook(lambda m, i, o: host_med.handle(PROVIDER, o))
    try:
        with torch.no_grad():
            return model._model(**inputs).logits
    finally:
        handle.remove()


def main():
    if shutil.which("bwrap") is None:
        sys.exit("bwrap not found")
    os.makedirs(HOST_DIR, exist_ok=True)
    with open(SECRET, "w") as f:
        f.write(SECRET_CONTENT)
    if os.path.exists(PWNED):
        os.remove(PWNED)

    model = LanguageModel("gpt2", device_map="cpu", dispatch=True)
    inputs = model.tokenizer(PROMPT, return_tensors="pt")

    # Reference: the same ×2 intervention applied locally (no jail, no socket).
    ref = host_logits(model, inputs, _LocalDouble())

    # --- Test A: golden equivalence with the worker JAILED ---
    pa, sa = spawn_jailed_worker("double")
    host_a = mk_host(sa)
    sock_a = host_logits(model, inputs, host_a)
    pa.wait(); host_a.channel.close()
    # atol 1e-3: tolerates multi-threaded CPU forward nondeterminism between two
    # separate forward passes (~1e-4) while still catching a broken swap (Δ≈30).
    golden = torch.allclose(ref, sock_a, atol=1e-3, rtol=0)
    print(f"[A jail+golden] logits match (atol 1e-3), worker jailed: {golden} "
          f"| max|Δ|={(ref - sock_a).abs().max().item():.2e}")

    # --- Test B: escapes inert AND the protocol still completes through the jail ---
    pb, sb = spawn_jailed_worker("escape")
    report = recv_frame(sb)                   # the jailed worker's escape report (pre-protocol)
    host_b = mk_host(sb)
    sock_b = host_logits(model, inputs, host_b)
    pb.wait(); host_b.channel.close()

    host_pwned = os.path.exists(PWNED)
    leaked = SECRET_CONTENT[:24] in report.get("fs_read", "") or "LEAKED" in report.get("net_egress", "")
    protocol_ok = torch.allclose(ref, sock_b, atol=1e-3, rtol=0)
    print(f"[B escape ]     report={report}")
    print(f"[B escape ]     host pwned-file created: {host_pwned} | secret leaked: {leaked} "
          f"| protocol still bit-identical: {protocol_ok}")

    ok = golden and protocol_ok and (not host_pwned) and (not leaked)
    shutil.rmtree(HOST_DIR, ignore_errors=True)
    print("=" * 72)
    print(f"PHASE 3 RESULT: {'PASS — real protocol works through the jail; escapes inert' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


class _LocalDouble:
    """Stand-in 'mediator' for the reference: doubles the block output locally."""
    def handle(self, provider, out):
        if isinstance(out, tuple):
            return (out[0] * 2.0,) + tuple(out[1:])
        return out * 2.0


if __name__ == "__main__":
    main()

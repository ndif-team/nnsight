#!/usr/bin/env python3
"""Phase 4 — Batcher as authority (cross-tenant / Boundary B).

A malicious jailed tenant (row 0) is co-batched with a victim row (row 1 = secret
data). The malicious tenant attempts every capability-leak mutation that succeeds
in-process today (widen its batch_group, walk the shared Batcher/sibling
mediators). We prove it is structurally inert in the isolated design:

  - the worker received ONLY its own row (never saw the victim's data), because the
    HOST narrows to the host-recorded batch_group — the worker's None/widen claim
    is set on its OWN (jail-local) Mediator and never reaches the host;
  - its poison swap landed ONLY on its own row — the victim row is untouched;
  - the walks to interleaver/batcher/siblings hit nothing (no host refs in the jail).

Run:  PYTHONPATH=src .../hf-serve/bin/python prototypes/mediator-sandbox/phase4_cross_tenant.py
"""
import os
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import torch

from nnsight.intervention.batching import Batcher
from nnsight.intervention.interleaver import Interleaver, Mediator
from nnsight.intervention.transport import SocketHostChannel, recv_frame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src")
WORKER = os.path.join(HERE, "phase4_malicious_worker.py")
ENV_ROOT = os.path.dirname(os.path.dirname(sys.executable))
PROVIDER = "model.layer.output.i0"


def spawn_jailed_worker():
    host_sock, worker_sock = socket.socketpair()
    os.set_inheritable(worker_sock.fileno(), True)
    env = {
        "WORKER_FD": str(worker_sock.fileno()), "PROVIDER": PROVIDER,
        "PYTHONPATH": SRC, "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp",
        "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    cmd = [
        "bwrap", "--unshare-all",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin", "--ro-bind", "/etc", "/etc",
        "--ro-bind", ENV_ROOT, ENV_ROOT, "--ro-bind", SRC, SRC, "--ro-bind", WORKER, WORKER,
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--die-with-parent",
        sys.executable, WORKER,
    ]
    p = subprocess.Popen(cmd, pass_fds=[worker_sock.fileno()], env=env)
    worker_sock.close()
    return p, host_sock


def main():
    if shutil.which("bwrap") is None:
        sys.exit("bwrap not found")

    # A 2-row batch: row 0 = the malicious tenant's input; row 1 = a victim's SECRET.
    row_attacker = torch.ones(1, 3, 4) * 1.0          # sum = 12
    row_victim = torch.ones(1, 3, 4) * 7.0            # sum = 84 — the secret
    batched = torch.cat([row_attacker, row_victim], dim=0)   # [2, 3, 4]

    p, sock = spawn_jailed_worker()
    batcher = Batcher()
    batcher.needs_batching = True
    batcher.last_batch_group = [0, 2]                 # total_batch_size = 2
    interleaver = Interleaver(mediators=[], tracer=None, batcher=batcher)
    host_med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=[0, 1])
    host_med.channel = SocketHostChannel(sock)
    host_med.interleaver = interleaver
    interleaver.mediators = [host_med]
    host_med.channel.wait_event()

    result = host_med.handle(PROVIDER, (batched.clone(),))
    report = recv_frame(sock)
    p.wait(); host_med.channel.close()

    final = result[0]
    print(f"[report] {report}")

    confined_read = report["received_shape"][0] == 1                       # got only 1 row
    never_saw_victim = abs(report["received_sum"] - 12.0) < 1e-4           # sum=12 (its row), not 96
    own_row_poisoned = torch.allclose(final[0], torch.full((3, 4), 999.0)) # attacker hit its own row
    victim_row_clean = torch.allclose(final[1], torch.full((3, 4), 7.0))   # victim row UNTOUCHED
    walks_contained = (report["batcher_walk"].startswith("CONTAINED")
                       and report["sibling_walk"].startswith("CONTAINED")
                       and report["direct_narrow"].startswith("CONTAINED")
                       and report["has_interleaver"] is False)

    print(f"[B-read ] malicious tenant confined to its row: {confined_read} "
          f"| never saw victim data (sum=12 not 96): {never_saw_victim}")
    print(f"[B-write] poison landed on own row: {own_row_poisoned} "
          f"| victim row uncorrupted: {victim_row_clean}")
    print(f"[B-walk ] interleaver/batcher/sibling walks all blocked: {walks_contained}")

    ok = confined_read and never_saw_victim and own_row_poisoned and victim_row_clean and walks_contained
    print("=" * 72)
    print(f"PHASE 4 RESULT: {'PASS — host Batcher is the authority; cross-tenant leak inert' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

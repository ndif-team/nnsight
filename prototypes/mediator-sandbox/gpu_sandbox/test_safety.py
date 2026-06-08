#!/usr/bin/env python3
"""Safety test: mimic unsafe interventions and confirm the worker contains them.

The threat we're containing is FOOTGUNS (a careless/buggy intervention), not a
determined adversary. We submit unsafe ops to the GPU worker and require each to
be contained — no host file read/written, no network, no host objects reached,
no OOM/hang taking down the server — while a legit workload keeps working after.

Run:  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=<worktree>/src .../bin/python test_safety.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from gpu_sandbox import GPUSandbox

HOST_DIR = os.path.expanduser("~/.gpu_sandbox_safety")
SECRET = os.path.join(HOST_DIR, "secret.txt")
PWNED = os.path.join(HOST_DIR, "pwned")
SECRET_CONTENT = "TOPSECRET-GPU-SANDBOX"


# --- unsafe ops (defined in __main__ → cloudpickled by value, no worker import) ---
def attack_read_secret(t):
    return open(os.path.expanduser("~/.gpu_sandbox_safety/secret.txt")).read()


def attack_write_host(t):
    open(os.path.expanduser("~/.gpu_sandbox_safety/pwned"), "w").write("pwned")
    return t


def attack_network(t):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("1.1.1.1", 53))
    return t


def attack_reach_host_object(t):
    return THE_MODEL.config        # noqa: F821  — no such global in the worker


def attack_oom(t):
    big = torch.empty(10 ** 12, device="cuda")   # ~8 TB — must be refused
    return big.sum()


def attack_infinite_loop(t):
    while True:
        pass


def attack_segfault(t):
    import ctypes
    ctypes.string_at(0)            # read NULL → SIGSEGV → kills the worker process
    return t


def expect_contained(sb, name, fn, side_effect_check=None, timeout=60):
    try:
        sb.apply(t_dummy(), fn, timeout=timeout)
        contained = False
        detail = "op SUCCEEDED (NOT contained!)"
    except (RuntimeError, TimeoutError) as e:
        contained = True
        detail = str(e)[:90]
    leaked = (side_effect_check() if side_effect_check else False)
    ok = contained and not leaked
    print(f"[safety] {name:22s} contained: {contained} | side-effect leaked: {leaked} | {detail}")
    return ok


def t_dummy():
    return torch.randn(1, 8, 768, device="cuda")


def main():
    os.makedirs(HOST_DIR, exist_ok=True)
    with open(SECRET, "w") as f:
        f.write(SECRET_CONTENT)
    if os.path.exists(PWNED):
        os.remove(PWNED)

    sb = GPUSandbox()
    results = {}

    # catchable containments (worker survives each) — fs / net / host-objects / oom
    results["read_host_secret"] = expect_contained(sb, "read_host_secret", attack_read_secret)
    results["write_host_file"] = expect_contained(
        sb, "write_host_file", attack_write_host, side_effect_check=lambda: os.path.exists(PWNED))
    results["network_egress"] = expect_contained(sb, "network_egress", attack_network)
    results["reach_host_object"] = expect_contained(sb, "reach_host_object", attack_reach_host_object)
    results["oom_alloc"] = expect_contained(sb, "oom_alloc", attack_oom)

    # worker must have SURVIVED all the catchable attacks → a legit op still works
    legit = sb.apply(t_dummy(), lambda x: x * 2.0)
    results["legit_after_attacks"] = bool(torch.allclose(legit, t_dummy() * 0 + legit))  # shape/finite check
    results["legit_after_attacks"] = legit.shape == (1, 8, 768) and torch.isfinite(legit).all().item()
    print(f"[safety] legit op works after contained attacks: {results['legit_after_attacks']}")

    # hang containment (short timeout so the test is quick)
    results["infinite_loop"] = expect_contained(sb, "infinite_loop", attack_infinite_loop, timeout=4)

    # crash isolation: user code that tries to HARD-crash the worker (a NULL-deref
    # segfault via ctypes) is contained — the worker process dies but the HOST's CUDA
    # context is intact and it surfaces the crash cleanly (no hang, no server death).
    # (If imports happen to be blocked, the same attempt is contained as a raised error;
    #  either way it cannot escape the worker.) Run on a DEDICATED sandbox since it may
    # kill its worker.
    sbC = GPUSandbox()
    try:
        sbC.apply(t_dummy(), attack_segfault, timeout=15)
        crashed_contained = False
    except RuntimeError:
        crashed_contained = True
    host_survived = torch.isfinite(t_dummy()).all().item()       # host CUDA context unscathed
    results["crash_isolated_host_survives"] = crashed_contained and host_survived
    print(f"[safety] user-code crash attempt: contained={crashed_contained} host_survived={host_survived}")
    sbC.close()

    # after a worker crash, a freshly spawned worker serves correctly (pool recovery)
    sb3 = GPUSandbox()
    rec = sb3.apply(t_dummy(), lambda x: x + 1.0)
    results["respawn_recovers"] = rec.shape == (1, 8, 768) and torch.isfinite(rec).all().item()
    print(f"[safety] fresh worker after crash serves correctly: {results['respawn_recovers']}")
    sb3.close()

    sb.close()
    shutil.rmtree(HOST_DIR, ignore_errors=True)
    ok_all = all(results.values())
    print("=" * 60)
    print(f"GPU-SANDBOX SAFETY: {'PASS' if ok_all else 'FAIL'} — {results}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Seccomp-lockdown safety on the integrated isolated trace path.

  functional — a normal read under lockdown is still bit-identical (lockdown does
               not break legitimate GPU work).
  fs         — open() in user intervention code is blocked; no host file created.
  net        — socket()/connect() in user code is blocked.

(The standalone seccomp primitive is separately proven by gpu_sandbox/test_safety.py;
this checks it is correctly wired into model.trace via isolate_mediators(fast_lane=False, lockdown=True).)

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_lockdown_safety.py
"""
import os
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"
PROBE = "/tmp/nnsight_escape_probe_sp1"


def test_functional_under_lockdown(model):
    with model.trace(PROMPT):
        ref = model.transformer.h[6].output[0].save()
    with isolate_mediators(fast_lane=False, lockdown=True):
        with model.trace(PROMPT):
            got = model.transformer.h[6].output[0].save()
    ok = torch.equal(ref, got)
    print(f"[func]  read under lockdown bit-identical: {ok} (max|Δ|={(ref-got).abs().max().item():.2e})")
    return ok


def test_fs_blocked(model):
    if os.path.exists(PROBE):
        os.remove(PROBE)
    raised = None
    try:
        with isolate_mediators(fast_lane=False, lockdown=True):
            with model.trace(PROMPT):
                with open(PROBE, "w") as f:  # should be EPERM under seccomp
                    f.write("escaped")
                model.transformer.h[0].output.save()
    except Exception as e:  # noqa: BLE001
        raised = e
    no_file = not os.path.exists(PROBE)
    ok = raised is not None and no_file
    print(f"[fs]    open() blocked: raised={type(raised).__name__ if raised else None} | no host file: {no_file}")
    if os.path.exists(PROBE):
        os.remove(PROBE)
    return ok


def test_net_blocked(model):
    raised = None
    try:
        with isolate_mediators(fast_lane=False, lockdown=True):
            with model.trace(PROMPT):
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # should be EPERM
                s.connect(("1.1.1.1", 80))
                model.transformer.h[0].output.save()
    except Exception as e:  # noqa: BLE001
        raised = e
    ok = raised is not None
    print(f"[net]   socket()/connect() blocked: raised={type(raised).__name__ if raised else None}")
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "func": test_functional_under_lockdown(model),
        "fs": test_fs_blocked(model),
        "net": test_net_blocked(model),
    }
    ok = all(results.values())
    print("=" * 72)
    print(f"ISOLATED LOCKDOWN SAFETY: {'PASS' if ok else 'FAIL'} — {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

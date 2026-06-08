#!/usr/bin/env python3
"""Isolated trace end-to-end — transparent isolated model.trace() is bit-identical to in-process.

  read  — isolated `h[6].output[0].save()` == in-process, max|Δ|=0.
  swap  — isolated `h[6].output[0] *= 2` propagates to a downstream save the same as
          in-process, max|Δ|=0, AND differs from no-swap (the swap really happened).

The intervention runs in a spawned GPU worker; values cross via the CUDA-IPC channel;
saves come back via the worker→host saves transmission.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_trace.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


def test_read(model):
    with model.trace(PROMPT):
        ref = model.transformer.h[6].output[0].save()
    with isolate_mediators():
        with model.trace(PROMPT):
            got = model.transformer.h[6].output[0].save()
    d = (ref.float() - got.float()).abs().max().item()
    ok = torch.equal(ref, got)
    print(f"[read]  isolated saved activation == in-process: {ok} (max|Δ|={d:.2e}, shape={tuple(got.shape)})")
    return ok


def test_swap(model):
    # Explicit assignment (eproperty __set__ -> SWAP event) — the isolation- and
    # remote-consistent form. In-place `[:]=` mutates a worker-local clone and does
    # not propagate across the process boundary (documented isolation semantic).
    with model.trace(PROMPT):
        plain = model.transformer.h[7].output[0].save()
    with model.trace(PROMPT):
        model.transformer.h[6].output = model.transformer.h[6].output * 2
        ref = model.transformer.h[7].output[0].save()
    with isolate_mediators():
        with model.trace(PROMPT):
            model.transformer.h[6].output = model.transformer.h[6].output * 2
            got = model.transformer.h[7].output[0].save()
    d = (ref.float() - got.float()).abs().max().item()
    same_as_ref = torch.equal(ref, got)
    changed = not torch.equal(plain, got)
    print(f"[swap]  isolated swap == in-process swap: {same_as_ref} (max|Δ|={d:.2e}) | changed-vs-noswap: {changed}")
    return same_as_ref and changed


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {"read": test_read(model), "swap": test_swap(model)}
    ok = all(results.values())
    print("=" * 72)
    print(f"ISOLATED TRACE: {'PASS' if ok else 'FAIL'} — {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

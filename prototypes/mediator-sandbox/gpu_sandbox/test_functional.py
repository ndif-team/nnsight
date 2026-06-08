#!/usr/bin/env python3
"""Functional test: real nnsight workloads still work, with the intervention op
offloaded to the isolated GPU worker.

For each real interpretability op we run a normal `model.trace()` two ways —
the op inline (reference) vs the op offloaded to the sandbox — and require the
final logits to match. The activation is delivered by nnsight's real machinery;
only the user's op runs in the isolated worker.

Run:  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=<worktree>/src .../bin/python test_functional.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from nnsight import LanguageModel
from gpu_sandbox import GPUSandbox

PROMPT = "The Eiffel Tower is in the city of"
LAYER = 6


# Real intervention ops (defined in __main__ so cloudpickle serializes them
# by value — the worker needs no import to rebuild them).
def scale(t):
    return t * 0.5


def steer(t):
    return t + 1.5


def ablate_mean(t):
    return t - t.mean(dim=-1, keepdim=True)


def project_norm(t):
    # a "read"-style op that changes shape: per-token L2 norm
    return t.norm(dim=-1)


def main():
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    sb = GPUSandbox()

    results = {}

    # 1. modify-activation ops: compare final logits, inline vs sandboxed
    for name, fn in {"scale": scale, "steer": steer, "ablate_mean": ablate_mean}.items():
        with model.trace(PROMPT):
            model.transformer.h[LAYER].output = fn(model.transformer.h[LAYER].output)
            ref = model.lm_head.output.save()
        with model.trace(PROMPT):
            h = model.transformer.h[LAYER].output
            model.transformer.h[LAYER].output = sb.apply(h, fn)
            out = model.lm_head.output.save()
        ok = torch.allclose(ref, out, atol=1e-3, rtol=0)
        results[name] = ok
        print(f"[functional] {name:12s} inline==sandboxed: {ok} | max|Δ|={(ref - out).abs().max():.2e}")

    # 2. read-style op that changes shape (norm per token): compare the read value
    with model.trace(PROMPT):
        ref_read = model.transformer.h[LAYER].output.norm(dim=-1).save()
    with model.trace(PROMPT):
        h = model.transformer.h[LAYER].output
        read = sb.apply(h, project_norm).save()
    ok = torch.allclose(ref_read, read, atol=1e-3, rtol=0)
    results["project_norm(read)"] = ok
    print(f"[functional] {'project_norm':12s} inline==sandboxed: {ok} | max|Δ|={(ref_read - read).abs().max():.2e}")

    # 3. the worker survives many requests (pool-readiness)
    reused = sb.alive()
    results["worker_alive_after_all"] = reused
    print(f"[functional] worker still alive after all requests: {reused}")

    sb.close()
    ok_all = all(results.values())
    print("=" * 60)
    print(f"GPU-SANDBOX FUNCTIONAL: {'PASS' if ok_all else 'FAIL'} — {results}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-token iteration isolated == in-process, bit-identical.

  steps  — iter[N] for N in {0,1,2}: saved activation at each step matches.
  swap   — swap h[6].output at step 1; downstream h[7] at step 1 matches in-process
           swap, and differs from no-swap.
  allsaved — iter[:] accumulating into a saved list (nnsight.save(hs)) matches per step.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_multitoken_iter.py
"""
import sys

import torch

import nnsight
from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"
N = 3


def test_steps(model):
    oks = []
    for step_n in range(N):
        with model.generate(PROMPT, max_new_tokens=N) as t:
            for step in t.iter[step_n]:
                ref = model.transformer.h[6].output[0].save()
        with isolate_mediators(timeout=30):
            with model.generate(PROMPT, max_new_tokens=N) as t:
                for step in t.iter[step_n]:
                    got = model.transformer.h[6].output[0].save()
        ok = torch.equal(ref, got)
        oks.append(ok)
        print(f"[steps] iter[{step_n}] match={ok} (max|Δ|={(ref-got).abs().max().item():.2e}, shape={tuple(got.shape)})", flush=True)
    return all(oks)


def test_swap(model):
    with model.generate(PROMPT, max_new_tokens=N) as t:
        for step in t.iter[1]:
            plain = model.transformer.h[7].output[0].save()
    with model.generate(PROMPT, max_new_tokens=N) as t:
        for step in t.iter[1]:
            model.transformer.h[6].output = model.transformer.h[6].output * 2
            ref = model.transformer.h[7].output[0].save()
    with isolate_mediators(timeout=30):
        with model.generate(PROMPT, max_new_tokens=N) as t:
            for step in t.iter[1]:
                model.transformer.h[6].output = model.transformer.h[6].output * 2
                got = model.transformer.h[7].output[0].save()
    ok = torch.equal(ref, got) and not torch.equal(plain, got)
    print(f"[swap]  iter[1] swap match={torch.equal(ref,got)} changed={not torch.equal(plain,got)}", flush=True)
    return ok


def test_allsaved(model):
    def run(iso):
        ctx = isolate_mediators(timeout=30) if iso else _null()
        with ctx:
            with model.generate(PROMPT, max_new_tokens=N) as t:
                hs = []
                for step in t.iter[:]:
                    hs.append(model.transformer.h[6].output[0])
                nnsight.save(hs)
        return hs
    ref = run(False)
    got = run(True)
    ok = isinstance(ref, list) and isinstance(got, list) and len(ref) == len(got) == N and all(
        torch.equal(a, b) for a, b in zip(ref, got)
    )
    print(f"[allsaved] iter[:] saved-list ref_n={len(ref) if isinstance(ref,list) else ref} got_n={len(got) if isinstance(got,list) else got} match={ok}", flush=True)
    return ok


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {}
    results["steps"] = test_steps(model)
    results["swap"] = test_swap(model)
    results["allsaved"] = test_allsaved(model)
    ok = all(results.values())
    print("=" * 72, flush=True)
    print(f"MULTI-TOKEN ITERATION: {'PASS' if ok else 'FAIL'} — {results}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

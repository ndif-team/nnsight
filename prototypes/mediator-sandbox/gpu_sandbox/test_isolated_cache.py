#!/usr/bin/env python3
"""tracer.cache() under isolation == in-process, bit-identical.

  one    — cache a single module (h[6]); cached output matches in-process.
  multi  — cache several modules; all entries match.
  inputs — include_inputs=True; inputs cached too.

Run:
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_cache.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


def _entry_out(cache, path):
    e = cache[path]
    e = e[-1] if isinstance(e, list) else e
    return e.output


def test_one(model):
    with model.trace(PROMPT) as t:
        ref = t.cache(modules=[model.transformer.h[6]]).save()
    with isolate_mediators(timeout=30):
        with model.trace(PROMPT) as t:
            got = t.cache(modules=[model.transformer.h[6]]).save()
    rk, gk = sorted(ref.keys()), sorted(got.keys())
    ok = rk == gk and len(gk) > 0 and torch.equal(
        _entry_out(ref, "transformer.h.6")[0], _entry_out(got, "transformer.h.6")[0]
    )
    print(f"[one]   keys ref={rk} got={gk} match={ok}", flush=True)
    return ok


def test_multi(model):
    mods = [model.transformer.h[2], model.transformer.h[5], model.transformer.h[9]]
    with model.trace(PROMPT) as t:
        ref = t.cache(modules=mods).save()
    with isolate_mediators(timeout=30):
        with model.trace(PROMPT) as t:
            got = t.cache(modules=mods).save()
    paths = ["transformer.h.2", "transformer.h.5", "transformer.h.9"]
    ok = sorted(ref.keys()) == sorted(got.keys()) and all(
        torch.equal(_entry_out(ref, p)[0], _entry_out(got, p)[0]) for p in paths
    )
    print(f"[multi] {len(got.keys())} keys match={ok}", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {"one": test_one(model), "multi": test_multi(model)}
    ok = all(results.values())
    print("=" * 72, flush=True)
    print(f"ISOLATED CACHE: {'PASS' if ok else 'FAIL'} — {results}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

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


def _entry(cache, path):
    e = cache[path]
    return e[-1] if isinstance(e, list) else e


def test_one(model):
    h6 = model.transformer.h[6]
    with model.trace(PROMPT) as t:
        ref = t.cache(modules=[h6]).save()
    with isolate_mediators(timeout=30):
        with model.trace(PROMPT) as t:
            got = t.cache(modules=[h6]).save()
    key = h6.path  # derive the key from the envoy path, don't hardcode the prefix
    rk, gk = sorted(ref.keys()), sorted(got.keys())
    ok = rk == gk and len(gk) > 0 and torch.equal(
        _entry(ref, key).output[0], _entry(got, key).output[0]
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
    paths = [m.path for m in mods]
    ok = sorted(ref.keys()) == sorted(got.keys()) and all(
        torch.equal(_entry(ref, p).output[0], _entry(got, p).output[0]) for p in paths
    )
    print(f"[multi] {len(got.keys())} keys match={ok}", flush=True)
    return ok


def test_inputs(model):
    # include_inputs=True: the cached module inputs must match in-process too.
    h4 = model.transformer.h[4]
    with model.trace(PROMPT) as t:
        ref = t.cache(modules=[h4], include_inputs=True).save()
    with isolate_mediators(timeout=30):
        with model.trace(PROMPT) as t:
            got = t.cache(modules=[h4], include_inputs=True).save()
    key = h4.path
    rk, gk = sorted(ref.keys()), sorted(got.keys())
    ok = rk == gk and len(gk) > 0 and torch.equal(
        _entry(ref, key).input, _entry(got, key).input
    )
    print(f"[inputs] keys match, input bit-identical={ok}", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "one": test_one(model),
        "multi": test_multi(model),
        "inputs": test_inputs(model),
    }
    ok = all(results.values())
    print("=" * 72, flush=True)
    print(f"ISOLATED CACHE: {'PASS' if ok else 'FAIL'} — {results}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

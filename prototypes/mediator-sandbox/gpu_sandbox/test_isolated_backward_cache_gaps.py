#!/usr/bin/env python3
"""backward/grad + cache() under isolation — characterizing the gaps.

  backward — get hidden + logits, then `with logits.sum().backward(): g = hidden.grad`.
  cache    — tracer.cache(modules=[...]) populated by hooks.

Each isolated-vs-in-process, hard timeout so deadlock shows as timeout.

Run:
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_backward_cache_gaps.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


def _run(fn):
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001
        return ("err", f"{type(e).__name__}: {str(e)[:120]}")


def test_backward(model):
    def body():
        with model.trace(PROMPT):
            hidden = model.transformer.h[6].output[0].save()
            logits = model.lm_head.output.save()
            with logits.sum().backward():
                g = hidden.grad.save()
        return g
    rs, rv = _run(body)
    def iso():
        with isolate_mediators(timeout=25):
            return body()
    gs, gv = _run(iso)
    ok = rs == "ok" and gs == "ok" and torch.is_tensor(rv) and torch.is_tensor(gv) and torch.equal(rv, gv)
    print(f"[backward] ref={rs} got={gs} match={ok if gs=='ok' else gv}", flush=True)
    return ok


def test_cache(model):
    def body():
        with model.trace(PROMPT) as t:
            cache = t.cache(modules=[model.transformer.h[6]]).save()
        return cache
    rs, rv = _run(body)
    def iso():
        with isolate_mediators(timeout=25):
            return body()
    gs, gv = _run(iso)
    # compare the cached output for h[6] if both ok
    ok = False
    if rs == "ok" and gs == "ok":
        try:
            rk = list(rv.keys()) if hasattr(rv, "keys") else rv
            gk = list(gv.keys()) if hasattr(gv, "keys") else gv
            ok = str(rk) == str(gk) and len(rk) > 0
        except Exception:
            ok = False
    print(f"[cache] ref={rs} got={gs} ok={ok} (gv={gv if gs!='ok' else 'CacheDict'})", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {"backward": test_backward(model), "cache": test_cache(model)}
    print("=" * 72, flush=True)
    print(f"BACKWARD/CACHE GAPS: {results}", flush=True)


if __name__ == "__main__":
    main()

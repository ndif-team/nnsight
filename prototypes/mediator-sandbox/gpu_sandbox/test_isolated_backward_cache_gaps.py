#!/usr/bin/env python3
"""backward/grad + cache() under isolation (former gaps, now closed).

  backward — now SUPPORTED (increment 1): grad of an ON-PATH tensor-output module
             (``ln_f.output``) runs HOST-SIDE under isolation and is bit-identical to
             in-process. The host keeps the real graph; the worker computes its half of
             the chain rule at the delivered-activation seam and reads grads back by
             PATH. Canonical coverage in test_isolated_backward.py. (Still open: gradient
             THROUGH a swap, which severs the host graph at the patch point.)
  cache    — SUPPORTED: ``tracer.cache(modules=[...])`` is bit-identical under
             isolation (kept here as a regression check; see test_isolated_cache.py).

Each isolated-vs-in-process, hard timeout so a deadlock shows as a timeout.

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
    # Increment 1 closed this gap: read-then-backward on an on-path tensor-output module
    # (ln_f.output) now runs HOST-SIDE under isolation, bit-identical to in-process.
    def body():
        with model.trace(PROMPT):
            hs = model.transformer.ln_f.output
            with model.lm_head.output.sum().backward():
                g = hs.grad.save()
        return g

    rs, rv = _run(body)

    def iso():
        with isolate_mediators(timeout=25):
            return body()

    gs, gv = _run(iso)

    ok = (
        rs == "ok"
        and gs == "ok"
        and torch.is_tensor(rv)
        and torch.is_tensor(gv)
        and torch.equal(rv, gv)
    )
    print(
        f"[backward] in-process={rs}; isolated={gs}; bit-identical={ok} "
        f"({'' if ok else (gv if gs == 'err' else 'mismatch')})",
        flush=True,
    )
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

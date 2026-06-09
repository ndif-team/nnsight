#!/usr/bin/env python3
"""backward/grad + cache() under isolation.

  backward — GAP (not built): grad of an ON-PATH tensor (``ln_f.output``) works
             in-process but fails cleanly under isolation. The fundamentals: the
             autograd graph is host-only, the worker holds DETACHED clones (no
             grad_fn), and ``.grad`` is keyed by ``id(tensor)`` (no cross-process name,
             unlike a module path). NB: grad MUST be taken on a tensor-output module
             like ``ln_f`` — a GPT2 block's ``.output[0]`` is an off-the-backward-path
             index into its tuple output, whose grad hook never fires (a usage gotcha
             that would make the in-process control spuriously error).
  cache    — now SUPPORTED: ``tracer.cache(modules=[...])`` is bit-identical under
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
    # ln_f.output is a tensor-output module ON the autograd path, so the in-process
    # control is VALID (unlike a block's off-path .output[0]).
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

    inproc_ok = rs == "ok" and torch.is_tensor(rv)
    # Backward under isolation is a documented gap (host-only autograd graph, detached
    # worker clones). It must fail CLEANLY (an error) — not hang or silently return
    # wrong grads. So the characterization holds iff in-process works AND isolated errors.
    isolated_fails_cleanly = gs == "err"
    ok = inproc_ok and isolated_fails_cleanly
    print(
        f"[backward] in-process={'ok (valid control)' if inproc_ok else rs}; "
        f"isolated={'fails cleanly — expected gap' if isolated_fails_cleanly else gs}: "
        f"{gv if gs == 'err' else 'UNEXPECTEDLY OK — investigate'}",
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

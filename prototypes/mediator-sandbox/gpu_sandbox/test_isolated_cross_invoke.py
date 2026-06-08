#!/usr/bin/env python3
"""Cross-invoke (barrier + variable sharing) under isolation — acceptance + characterization.

  xinvoke — invoke A captures a var; invoke B uses it (cross_invoker var-sharing).
  barrier — the canonical tracer.barrier(2) cross-invoke embeddings-copy pattern.

Each isolated-vs-in-process, with a hard timeout so deadlock shows as timeout.

Run:
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_cross_invoke.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

A = "The Eiffel Tower is in the city of"
B = "The capital of France is the city of"


def _run(fn):
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001
        return ("err", f"{type(e).__name__}: {str(e)[:100]}")


def test_xinvoke(model):
    # invoke B reads a tensor variable defined in invoke A, with a barrier to order it.
    def body():
        with model.trace() as t:
            bar = t.barrier(2)
            with t.invoke(A):
                captured = model.transformer.h[3].output
                bar()
                a_out = model.lm_head.output.save()
            with t.invoke(B):
                bar()
                model.transformer.h[3].output = captured  # cross-invoke use
                b_out = model.lm_head.output.save()
        return a_out, b_out

    rs, rv = _run(body)
    def iso():
        with isolate_mediators(timeout=20):
            return body()
    gs, gv = _run(iso)
    ok = rs == "ok" and gs == "ok" and torch.equal(rv[0], gv[0]) and torch.equal(rv[1], gv[1])
    print(f"[xinvoke] ref={rs} got={gs} match={ok if gs=='ok' else gv}", flush=True)
    return ok


def test_barrier_only(model):
    # Two invokes that both just hit a barrier (no var sharing) — pure sync.
    def body():
        with model.trace() as t:
            bar = t.barrier(2)
            with t.invoke(A):
                bar()
                a = model.transformer.h[2].output[0].save()
            with t.invoke(B):
                bar()
                b = model.transformer.h[2].output[0].save()
        return a, b

    rs, rv = _run(body)
    def iso():
        with isolate_mediators(timeout=20):
            return body()
    gs, gv = _run(iso)
    ok = (
        rs == "ok" and gs == "ok"
        and torch.equal(rv[0], gv[0]) and torch.equal(rv[1], gv[1])
    )
    print(f"[barrier] ref={rs} got={gs} match={ok if gs=='ok' else gv}", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {}
    results["barrier"] = test_barrier_only(model)
    results["xinvoke"] = test_xinvoke(model)
    print("=" * 72, flush=True)
    print(f"CROSS-INVOKE (barrier + variable sharing): {results}", flush=True)


if __name__ == "__main__":
    main()

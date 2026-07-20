#!/usr/bin/env python3
"""Increment 1 — read-then-backward under isolation == in-process, bit-identical.

Scope (the simplest correct slice): single invoke, NO swaps, the gradient target is
an ON-PATH tensor-output module (``ln_f.output``), the loss is a scalar reduction of a
read activation. The host graph stays intact (a read does not replace the host's real
tensor), so the host can run the real backward; the worker only computes the seed
gradient at the loss's delivered leaves and reads grads back by PATH.

  read_backward — grad of ln_f.output w.r.t. a sum-of-logits loss; max|Δ|=0 vs in-process.

NOT in scope here (later increments): gradient through a SWAP (severs the host graph),
tuple-element targets (off-path), batched/multi-invoke backward.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_backward.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


def _read_backward(model):
    # ln_f.output is a tensor-output module ON the autograd path.
    with model.trace(PROMPT):
        hs = model.transformer.ln_f.output
        with model.lm_head.output.sum().backward():
            g = hs.grad.save()
    return g


def test_read_backward(model):
    ref = _read_backward(model)
    with isolate_mediators(fast_lane=False, timeout=30):
        got = _read_backward(model)

    ok = (
        torch.is_tensor(ref)
        and torch.is_tensor(got)
        and ref.shape == got.shape
        and torch.equal(ref, got)
    )
    delta = (ref - got).abs().max().item() if (torch.is_tensor(ref) and torch.is_tensor(got) and ref.shape == got.shape) else float("nan")
    print(
        f"[read_backward] ref={tuple(ref.shape) if torch.is_tensor(ref) else ref} "
        f"got={tuple(got.shape) if torch.is_tensor(got) else got} "
        f"max|Δ|={delta} match={ok}",
        flush=True,
    )
    return ok


def test_read_backward_nonstd():
    # Testing rule: vary names, don't assume GPT-2 conventions. Rename the norm + head
    # to non-standard user-facing paths; the host must resolve grads by the real path.
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)

    def body():
        with model.trace(PROMPT):
            hs = model.final_norm.output
            with model.output_projection.output.sum().backward():
                g = hs.grad.save()
        return g

    ref = body()
    with isolate_mediators(fast_lane=False, timeout=30):
        got = body()
    ok = torch.is_tensor(ref) and torch.is_tensor(got) and torch.equal(ref, got)
    delta = (ref - got).abs().max().item() if ok or (torch.is_tensor(ref) and torch.is_tensor(got)) else float("nan")
    print(f"[nonstd] final_norm.output grad isolated==in-proc: {ok} (max|Δ|={delta})", flush=True)
    return ok


def test_derived_target_fails_clean(model):
    # Boundary: gradient of a USER-DERIVED tensor has no host-side graph under isolation.
    # It must fail CLEANLY (a clear error), never hang or return a silently-wrong grad.
    def body():
        with model.trace(PROMPT):
            hs = model.transformer.ln_f.output
            derived = hs * 2  # computed in user code — no module-path provenance
            with model.lm_head.output.sum().backward():
                g = derived.grad.save()
        return g

    try:
        body()
        print("[derived] UNEXPECTEDLY OK — should have refused a derived-tensor grad", flush=True)
        return False
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        ok = "isolation" in msg.lower()
        print(f"[derived] failed cleanly={ok}: {type(e).__name__}: {msg[:110]}", flush=True)
        return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "read_backward": test_read_backward(model),
        "nonstd": test_read_backward_nonstd(),
        "derived_fails_clean": _derived_isolated(model),
    }
    print("=" * 72, flush=True)
    print(f"ISOLATED BACKWARD (increment 1): {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


def _derived_isolated(model):
    with isolate_mediators(fast_lane=False, timeout=30):
        return test_derived_target_fails_clean(model)


if __name__ == "__main__":
    main()

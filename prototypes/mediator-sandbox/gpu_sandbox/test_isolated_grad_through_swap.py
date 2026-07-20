#!/usr/bin/env python3
"""Gradient THROUGH a swap under isolation == in-process, bit-identical.

A SWAP (replacement write) installs a worker-computed value on the host as a detached
*leaf*, so the host autograd graph is severed at the swap seam: a downstream loss
differentiated w.r.t. an UPSTREAM activation dead-ends at the swap (it returns no gradient),
while in-process the gradient flows through the swap. The fix stitches the seam: the host
returns dL/d(swap leaf), the worker backprops through its swap tape to dL/d(delivered clone),
and the host continues the pre-swap backward — iterated to a fixpoint, grads summed across
rounds (a clone reached both directly and through a swap contributes via both paths).

  mul        — swap h[6].output := 2*h[6].output; grad of an upstream h[3].output matches
               in-process (the *2 factor propagates back through the seam).
  add        — swap := h[6].output + vec (additive, like steering); grad of h[3] matches.
  steer      — tracer.steer at h[6], then backward; grad of h[3] matches (steer is a swap).
  two_swaps  — swaps at h[4] AND h[7]; grad of h[2] flows through BOTH seams (loop fixpoint).
  renamed    — renamed model (decoder_blocks): grad through a swap matches in-process.

Run:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/nnsight-tf/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_grad_through_swap.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

P = "The Eiffel Tower is in the city of"


def U(o):
    return o[0] if isinstance(o, tuple) else o


def _eq(a, b):
    return (torch.is_tensor(a) and torch.is_tensor(b)
            and a.shape == b.shape and torch.equal(a, b))


def _delta(a, b):
    return (a - b).abs().max().item() if _eq(a, b) else float("nan")


def _both(build):
    ref = build()
    with isolate_mediators(fast_lane=False, timeout=30):
        got = build()
    return ref, got


def test_mul(model):
    def build():
        with model.trace(P):
            up = U(model.transformer.h[3].output)
            mid = U(model.transformer.h[6].output)
            model.transformer.h[6].output = mid * 2.0
            loss = model.lm_head.output.sum()
            with loss.backward():
                g = up.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[mul]   grad through h6:=2*h6, wrt h3, isolated==in-process={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_add(model):
    vec = torch.randn(768)

    def build():
        with model.trace(P):
            up = U(model.transformer.h[3].output)
            mid = U(model.transformer.h[6].output)
            model.transformer.h[6].output = mid + vec.to(dtype=mid.dtype, device=mid.device)
            loss = model.lm_head.output.sum()
            with loss.backward():
                g = up.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[add]   grad through h6:=h6+vec, wrt h3, isolated==in-process={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_steer(model):
    direction = torch.randn(768)

    def build():
        with model.trace(P) as tracer:
            up = U(model.transformer.h[3].output)
            tracer.steer(model.transformer.h[6], direction, 4.0)
            loss = model.lm_head.output.sum()
            with loss.backward():
                g = up.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[steer] grad through tracer.steer(h6), wrt h3, isolated==in-process={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_two_swaps(model):
    def build():
        with model.trace(P):
            up = U(model.transformer.h[2].output)
            m4 = U(model.transformer.h[4].output)
            model.transformer.h[4].output = m4 * 1.5
            m7 = U(model.transformer.h[7].output)
            model.transformer.h[7].output = m7 * 0.7
            loss = model.lm_head.output.sum()
            with loss.backward():
                g = up.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[two_swaps] grad through h4 & h7 swaps, wrt h2, isolated==in-process={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_renamed():
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection",
              "transformer.h": "decoder_blocks"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)

    def build():
        with model.trace(P):
            up = U(model.decoder_blocks[3].output)
            mid = U(model.decoder_blocks[6].output)
            model.decoder_blocks[6].output = mid * 2.0
            loss = model.output_projection.output.sum()
            with loss.backward():
                g = up.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[renamed] grad through swap (renamed), isolated==in-process={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "mul": test_mul(model),
        "add": test_add(model),
        "steer": test_steer(model),
        "two_swaps": test_two_swaps(model),
        "renamed": test_renamed(),
    }
    print("=" * 72, flush=True)
    print(f"ISOLATED GRAD-THROUGH-SWAP: {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

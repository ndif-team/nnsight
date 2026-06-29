#!/usr/bin/env python3
"""Batched backward under isolation == in-process, bit-identical.

"Batched" here is a LIST input (`model.trace([A, B, ...])`): one mediator, one forward over
the padded batch, one backward, per-row gradients. There is no per-invoke narrowing (the
mediator's batch_group is None), so the worker's delivered clone and the host's retained real
are both full-batch and their shapes match: the read-path and grad-through-swap seam-stitch
work unchanged, just on a (batch, seq, hidden) tensor.

(Backward inside MULTIPLE `tracer.invoke(...)` contexts is a separate, unsupported structure:
it raises `MissedProviderError` IN-PROCESS too, so it is a core nnsight limitation, not an
isolation gap; not covered here.)

  two_rows   — grad of ln_f.output for a 2-prompt batch, isolated == in-process, shape (2,·,768).
  three_rows — same for a 3-prompt batch.
  upstream   — grad of an upstream block (h[3].output) in a batched trace.
  swap       — batched + a swap at h[6]; grad of upstream h[3] flows through the seam (batched
               grad-through-swap), isolated == in-process.
  renamed    — renamed model (decoder_blocks), batched backward.

Run:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/nnsight-tf/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_batched_backward.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

A = "The Eiffel Tower is in the city of"
B = "A red bicycle was left near the river of"
C = "She quietly closed the heavy wooden front"


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


def test_two_rows(model):
    def build():
        with model.trace([A, B]):
            hs = U(model.transformer.ln_f.output)
            with model.lm_head.output.sum().backward():
                g = hs.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got) and ref.shape[0] == 2
    print(f"[two_rows]   batched grad bit-identical={_eq(ref, got)} shape={tuple(ref.shape)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_three_rows(model):
    def build():
        with model.trace([A, B, C]):
            hs = U(model.transformer.ln_f.output)
            with model.lm_head.output.sum().backward():
                g = hs.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got) and ref.shape[0] == 3
    print(f"[three_rows] batched grad bit-identical={_eq(ref, got)} shape={tuple(ref.shape)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_upstream(model):
    def build():
        with model.trace([A, B]):
            up = U(model.transformer.h[3].output)
            with model.lm_head.output.sum().backward():
                g = up.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[upstream]   batched grad of h[3] bit-identical={ok} shape={tuple(ref.shape)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_swap(model):
    def build():
        with model.trace([A, B]):
            up = U(model.transformer.h[3].output)
            mid = U(model.transformer.h[6].output)
            model.transformer.h[6].output = mid * 2.0
            loss = model.lm_head.output.sum()
            with loss.backward():
                g = up.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[swap]       batched grad-through-swap bit-identical={ok} shape={tuple(ref.shape)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_renamed():
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection",
              "transformer.h": "decoder_blocks"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)

    def build():
        with model.trace([A, B]):
            hs = U(model.final_norm.output)
            with model.output_projection.output.sum().backward():
                g = hs.grad.save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got) and ref.shape[0] == 2
    print(f"[renamed]    batched grad (renamed) bit-identical={_eq(ref, got)} shape={tuple(ref.shape)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "two_rows": test_two_rows(model),
        "three_rows": test_three_rows(model),
        "upstream": test_upstream(model),
        "swap": test_swap(model),
        "renamed": test_renamed(),
    }
    print("=" * 72, flush=True)
    print(f"ISOLATED BATCHED BACKWARD: {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

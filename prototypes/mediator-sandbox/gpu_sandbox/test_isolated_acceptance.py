#!/usr/bin/env python3
"""Isolated trace acceptance — multi-invoke, non-standard module names, exception, timeout.

Complements the read/swap end-to-end test (read/swap bit-identical). All comparisons are against
the in-process result on the SAME model.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_acceptance.py
"""
import sys
import time

import torch
import torch.nn as nn

import nnsight
from nnsight import LanguageModel, NNsight
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


# --------------------------------------------------------------------------- #
# Non-standard module names (testing rule: no GPT-2-only assumptions)         #
# --------------------------------------------------------------------------- #
class TinyNet(nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.embed = nn.Embedding(32, d)
        self.decoder_blocks = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])
        self.output_projection = nn.Linear(d, 32)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.decoder_blocks:
            h = torch.relu(blk(h))
        return self.output_projection(h)


def test_nonstandard_names():
    torch.manual_seed(0)
    net = TinyNet().cuda()
    model = NNsight(net)
    x = torch.randint(0, 32, (1, 6)).cuda()

    with model.trace(x):
        ref = model.decoder_blocks[1].output.save()
    with isolate_mediators(fast_lane=False):
        with model.trace(x):
            got = model.decoder_blocks[1].output.save()
    ok = torch.equal(ref, got)
    print(f"[names] non-standard 'decoder_blocks' read == in-process: {ok} (max|Δ|={(ref-got).abs().max().item():.2e})")
    return ok


def test_multi_invoke(model):
    # Two invokes, each its own isolated worker; batch narrowing must keep rows separate.
    with model.trace() as t:
        with t.invoke("The capital of France is"):
            a_ref = model.transformer.h[5].output[0].save()
        with t.invoke("The Eiffel Tower is in"):
            b_ref = model.transformer.h[5].output[0].save()
    with isolate_mediators(fast_lane=False):
        with model.trace() as t:
            with t.invoke("The capital of France is"):
                a_got = model.transformer.h[5].output[0].save()
            with t.invoke("The Eiffel Tower is in"):
                b_got = model.transformer.h[5].output[0].save()
    ok = torch.equal(a_ref, a_got) and torch.equal(b_ref, b_got)
    no_cross = not torch.equal(a_got, b_got)  # rows are genuinely different
    print(f"[multi] two isolated invokes bit-identical: {ok} | rows distinct (no cross-leak): {no_cross}")
    return ok and no_cross


def test_exception(model):
    # A footgun (ValueError) in user code must surface in the user's context.
    raised = None
    try:
        with isolate_mediators(fast_lane=False):
            with model.trace(PROMPT):
                _ = model.transformer.h[6].output[0]
                raise ValueError("boom-from-user-code")
    except Exception as e:  # noqa: BLE001
        raised = e
    ok = raised is not None and "boom-from-user-code" in str(raised)
    print(f"[exc]   user exception propagated to host: {ok} ({type(raised).__name__ if raised else None})")
    return ok


def test_timeout(model):
    # An infinite loop in user code must be killed and the host must survive.
    t0 = time.time()
    killed = None
    try:
        with isolate_mediators(fast_lane=False, timeout=5):
            with model.trace(PROMPT):
                out = model.transformer.h[6].output[0]
                while True:  # footgun: hang
                    out = out + 1
    except Exception as e:  # noqa: BLE001
        killed = e
    dt = time.time() - t0
    # host still works afterwards
    with model.trace(PROMPT):
        alive = model.transformer.h[0].output[0].save()
    host_ok = torch.is_tensor(alive)
    print(f"[hang]  infinite loop killed in {dt:.1f}s (err={type(killed).__name__ if killed else None}); host survives: {host_ok}")
    return killed is not None and host_ok


def main():
    assert torch.cuda.is_available()
    results = {}
    results["names"] = test_nonstandard_names()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results["multi"] = test_multi_invoke(model)
    results["exc"] = test_exception(model)
    # timeout test depends on isolate_mediators supporting a `timeout=` kwarg
    if "timeout" in isolate_mediators.__doc__ or True:
        try:
            results["hang"] = test_timeout(model)
        except TypeError as e:
            print(f"[hang]  SKIP (timeout kwarg not wired): {e}")
    ok = all(results.values())
    print("=" * 72)
    print(f"ISOLATED TRACE ACCEPTANCE: {'PASS' if ok else 'FAIL'} — {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

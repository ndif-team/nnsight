#!/usr/bin/env python3
"""Non-standard module names under isolation (testing rule: vary names, not GPT-2-only).

generate() needs a real HF generative model, so we use gpt2 with rename= to give it
non-standard USER-FACING paths (decoder_blocks / output_projection). This catches any
alias-path vs real-path mismatch in the host-side hook registration's requester->envoy resolution.

  read   — model.decoder_blocks[6].output[0].save() isolated == in-process.
  iterN  — generate + iter[1] on the renamed path, isolated == in-process.

Run:
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_nonstd.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"
RENAME = {"transformer.h": "decoder_blocks", "lm_head": "output_projection"}


def test_read(model):
    with model.trace(PROMPT):
        ref = model.decoder_blocks[6].output[0].save()
    with isolate_mediators(fast_lane=False, timeout=30):
        with model.trace(PROMPT):
            got = model.decoder_blocks[6].output[0].save()
    ok = torch.equal(ref, got)
    print(f"[read]  decoder_blocks[6] isolated==in-proc: {ok} (max|Δ|={(ref-got).abs().max().item():.2e})", flush=True)
    return ok


def test_iterN(model):
    with model.generate(PROMPT, max_new_tokens=3) as t:
        for step in t.iter[1]:
            ref = model.decoder_blocks[6].output[0].save()
    with isolate_mediators(fast_lane=False, timeout=30):
        with model.generate(PROMPT, max_new_tokens=3) as t:
            for step in t.iter[1]:
                got = model.decoder_blocks[6].output[0].save()
    ok = torch.equal(ref, got)
    print(f"[iterN] decoder_blocks[6] iter[1] isolated==in-proc: {ok} (max|Δ|={(ref-got).abs().max().item():.2e})", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=RENAME)
    results = {"read": test_read(model), "iterN": test_iterN(model)}
    ok = all(results.values())
    print("=" * 72, flush=True)
    print(f"NONSTD (renamed paths): {'PASS' if ok else 'FAIL'} — {results}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Host-routed unembed on the ISOLATED tier: tracer.unembed runs the real final-norm +
unembed on the host's real weights even when the intervention runs in the weightless
worker — so the weight-reading interp readout works under forced isolation, not only on
the fast lane.

All cases force isolation (fast_lane=False) so the cell genuinely runs in the worker and
exercises the worker -> Events.UNEMBED -> host handler -> logits-back round trip.

  single        — one layer's residual projected via tracer.unembed, isolated ==
                  in-process, bit-identical.
  multi         — three layers (forward order) interleaved with unembed, bit-identical.
  module_form   — formulation="module" (head(normed)) bit-identical.
  no_norm       — norm=None (skip normalization) bit-identical.
  renamed       — renamed model (final_norm / output_projection / decoder_blocks):
                  paths resolve host-side, bit-identical (no hardcoded names).
  matches_raw   — tracer.unembed equals the manual F.linear(norm(x), head.weight).

Run:
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_unembed.py
"""
import sys

import torch
import torch.nn.functional as F

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


def _both(build):
    """Run build() in-process and under forced isolation; return (ref, got)."""
    ref = build()
    with isolate_mediators(fast_lane=False, timeout=30):
        got = build()
    return ref, got


def _eq(ref, got):
    return (torch.is_tensor(ref) and torch.is_tensor(got)
            and ref.shape == got.shape and torch.equal(ref, got))


def _delta(ref, got):
    return (ref - got).abs().max().item() if _eq(ref, got) else float("nan")


def test_single(model):
    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                resid = model.transformer.h[6].output
                logits = tracer.unembed(resid, model.transformer.ln_f, model.lm_head)
                row = logits[:, -1, :].save()
        return row
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[single] isolated unembed bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_multi(model):
    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                rows = []
                for i in [0, 4, 8]:
                    resid = model.transformer.h[i].output
                    rows.append(tracer.unembed(resid, model.transformer.ln_f,
                                               model.lm_head)[:, -1, :])
                g = torch.stack(rows, dim=0).save()
        return g
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[multi] 3-layer isolated unembed bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_module_form(model):
    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                resid = model.transformer.h[6].output
                logits = tracer.unembed(resid, model.transformer.ln_f, model.lm_head,
                                        formulation="module")
                row = logits[:, -1, :].save()
        return row
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[module_form] formulation='module' bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_no_norm(model):
    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                resid = model.transformer.h[6].output
                logits = tracer.unembed(resid, None, model.lm_head)  # skip norm
                row = logits[:, -1, :].save()
        return row
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[no_norm] norm=None bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_renamed():
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection",
              "transformer.h": "decoder_blocks"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                resid = model.decoder_blocks[5].output
                logits = tracer.unembed(resid, model.final_norm, model.output_projection)
                row = logits[:, -1, :].save()
        return row
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[renamed] renamed-model isolated unembed bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_matches_raw(model):
    # tracer.unembed (in-process) must equal the manual readout it replaces.
    with model.trace(PROMPT) as tracer:
        with torch.no_grad():
            resid = model.transformer.h[6].output
            via_api = tracer.unembed(resid, model.transformer.ln_f, model.lm_head)[:, -1, :].save()
    with model.trace(PROMPT):
        with torch.no_grad():
            out = model.transformer.h[6].output
            hidden = out[0] if isinstance(out, tuple) else out
            manual = F.linear(model.transformer.ln_f(hidden), model.lm_head.weight)[:, -1, :].save()
    ok = _eq(via_api, manual)
    print(f"[matches_raw] tracer.unembed == manual F.linear bit-identical={ok} "
          f"(max|Δ|={_delta(via_api, manual)})", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "single": test_single(model),
        "multi": test_multi(model),
        "module_form": test_module_form(model),
        "no_norm": test_no_norm(model),
        "renamed": test_renamed(),
        "matches_raw": test_matches_raw(model),
    }
    print("=" * 72, flush=True)
    print(f"ISOLATED UNEMBED: {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

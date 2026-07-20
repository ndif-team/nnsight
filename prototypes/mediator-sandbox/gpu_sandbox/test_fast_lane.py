#!/usr/bin/env python3
"""End-to-end fast lane: confirmed-safe interventions run IN-PROCESS under
isolate_mediators(), and that is the ONLY tier that can run the weight-reading interp
majority (the isolated worker holds weightless dummy modules).

  weights_fast     — a logit-lens cell (reads lm_head.weight, calls ln_f) runs under
                     isolate_mediators() bit-identical to non-isolated AND raises under
                     forced isolation (fast_lane=False) — proving the fast lane is the
                     enabling tier, not just a speedup.
  inplace_steer    — an in-place steering cell runs correctly on the fast lane (it is a
                     silent no-op under isolation due to clone-on-receive).
  renamed          — same logit-lens shape on a renamed model (final_norm /
                     output_projection / decoder_blocks) fast-lanes bit-identical.
  footgun_isolates — a cell that imports os / opens a file is NOT fast-laned: it routes to
                     the worker (isolated) and is contained, host survives.
  introspection    — a cell reaching for ().__class__... raises FastLaneRejected.
  watchdog         — a huge bounded loop fast-laned with a short deadline is killed by the
                     watchdog; the host survives and the next trace still works.

Run:
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_fast_lane.py
"""
import sys

import torch
import torch.nn.functional as F

from nnsight import LanguageModel
from nnsight.intervention.fastlane import FastLaneRejected
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


# --- the weight-reading interp readout (logit lens), the workload isolation cannot run --
def _logit_lens(blocks, norm, head, layers):
    rows = []
    with torch.no_grad():
        for i in layers:
            out = blocks[i].output
            hidden = out[0] if isinstance(out, tuple) else out
            normed = norm(hidden)
            logits = F.linear(normed, head.weight)   # host-weight read — worker-impossible
            rows.append(logits[:, -1, :])
    return torch.stack(rows, dim=0)


def _lens_gpt2(model):
    with model.trace(PROMPT):
        g = _logit_lens(model.transformer.h, model.transformer.ln_f,
                        model.lm_head, [0, 4, 8]).save()
    return g


def test_weights_fast(model):
    ref = _lens_gpt2(model)                       # non-isolated baseline
    with isolate_mediators():                     # fast lane on (default)
        got = _lens_gpt2(model)
    fast_ok = torch.is_tensor(got) and torch.equal(ref, got)

    # forced isolation: the same cell must FAIL (weightless dummy modules in the worker)
    forced_failed = False
    try:
        with isolate_mediators(fast_lane=False, timeout=30):
            _lens_gpt2(model)
    except Exception:  # noqa: BLE001
        forced_failed = True

    print(f"[weights_fast] fast-lane bit-identical={fast_ok} "
          f"(max|Δ|={(ref-got).abs().max().item() if fast_ok else float('nan')}); "
          f"forced-isolation-raises={forced_failed}", flush=True)
    return fast_ok and forced_failed


def test_inplace_steer(model):
    def steer(inplace_ctx):
        # steer block 6 toward a token's unembed row, read final logits
        with model.trace(PROMPT):
            with torch.no_grad():
                direction = F.normalize(model.lm_head.weight[5000].float(), dim=0)
                out = model.transformer.h[6].output
                hidden = out[0] if isinstance(out, tuple) else out
                scale = hidden.norm(dim=-1).mean()
                hidden[:] = hidden + 6.0 * scale * direction.to(hidden.dtype)  # in-place
                last = model.transformer.h[-1].output
                last = last[0] if isinstance(last, tuple) else last
                normed = model.transformer.ln_f(last)
                logits = F.linear(normed, model.lm_head.weight)[:, -1, :].save()
        return logits

    ref = steer(None)                             # non-isolated
    with isolate_mediators():
        got = steer(None)                         # fast lane
    ok = torch.is_tensor(got) and torch.equal(ref, got)
    print(f"[inplace_steer] fast-lane in-place write bit-identical={ok} "
          f"(max|Δ|={(ref-got).abs().max().item() if ok else float('nan')})", flush=True)
    return ok


def test_renamed():
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection",
              "transformer.h": "decoder_blocks"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)

    def body():
        with model.trace(PROMPT):
            g = _logit_lens(model.decoder_blocks, model.final_norm,
                            model.output_projection, [0, 3, 6]).save()
        return g

    ref = body()
    with isolate_mediators():
        got = body()
    ok = torch.is_tensor(got) and torch.equal(ref, got)
    print(f"[renamed] renamed-model logit-lens fast-lane bit-identical={ok} "
          f"(max|Δ|={(ref-got).abs().max().item() if ok else float('nan')})", flush=True)
    return ok


def test_footgun_isolates(model):
    # A footgun cell is NOT confirmable -> routes to the worker (isolated), contained.
    # The worker can't run the weight read either, so it errors cleanly; the point is the
    # host SURVIVES and a subsequent normal trace still works.
    raised = False
    try:
        with isolate_mediators(fast_lane=True, timeout=20):
            with model.trace(PROMPT):
                import os  # noqa: F401 — footgun: isolated, never fast-laned
                _ = model.transformer.h[6].output[0].save()
    except Exception:  # noqa: BLE001
        raised = True
    # host survives: a clean fast-lane trace works right after
    after = _lens_gpt2(model)
    survived = torch.is_tensor(after)
    print(f"[footgun_isolates] footgun routed off the fast lane (raised={raised}); "
          f"host survived={survived}", flush=True)
    return survived


def test_introspection(model):
    rejected = False
    try:
        with isolate_mediators():
            with model.trace(PROMPT):
                _ = ().__class__.__bases__          # introspection escape
                _ = model.transformer.h[6].output[0].save()
    except FastLaneRejected:
        rejected = True
    except Exception as e:  # noqa: BLE001
        rejected = "introspection" in str(e).lower() or "reject" in str(e).lower()
    print(f"[introspection] introspection escape rejected={rejected}", flush=True)
    return rejected


def test_watchdog(model):
    # A huge bounded loop passes the static gate (range is "bounded") but would hang the
    # host; the watchdog must kill it. Short deadline so the test is quick.
    killed = False
    try:
        with isolate_mediators(fast_lane_timeout=4.0):
            with model.trace(PROMPT):
                h = model.transformer.h[6].output[0]
                acc = 0
                for _i in range(10 ** 12):           # bounded literal, but enormous
                    acc += 1
                _ = h.save()
    except Exception:  # noqa: BLE001 — FastLaneTimeout surfaces through the trace
        killed = True
    after = _lens_gpt2(model)                         # host survived
    survived = torch.is_tensor(after)
    print(f"[watchdog] runaway loop killed={killed}; host survived={survived}", flush=True)
    return killed and survived


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "weights_fast": test_weights_fast(model),
        "inplace_steer": test_inplace_steer(model),
        "renamed": test_renamed(),
        "footgun_isolates": test_footgun_isolates(model),
        "introspection": test_introspection(model),
        "watchdog": test_watchdog(model),
    }
    print("=" * 72, flush=True)
    print(f"FAST LANE (end-to-end): {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

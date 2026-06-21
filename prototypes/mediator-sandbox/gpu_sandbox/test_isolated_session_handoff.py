#!/usr/bin/env python3
"""Cross-trace (run<->run) handoff inside model.session() on the ISOLATED tier.

In-process, a session carries values across its inner traces: a value produced in trace 1
is visible in trace 2 (saved or not), because each inner trace pushes its locals up to the
session frame. Under isolation each inner trace runs in a worker that ships only its
``.save()``'d values home, so cross-trace handoff broke two ways:
  - SAVED value used cross-trace -> the host wrote it to the session frame but never
    re-registered its host id in Globals.saves, so the session's exit-push dropped it
    (UnboundLocalError).
  - NON-saved value used cross-trace -> the worker never shipped it (NameError).

This verifies both are fixed:
  - the documented ``hs = x.save()`` -> use ``hs`` session pattern works under isolation;
  - ``x.carry()`` explicitly hands a value to a later trace WITHOUT surfacing it as a saved
    output, and is portable (in-process == isolated).

  saved_handoff  — session, trace1 ``hs = x.save()``, trace2 patches ``hs*1.5`` -> downstream
                   bit-identical isolated vs in-process, and changed vs the unpatched baseline.
  carry_handoff  — same but ``hs = x.carry()`` (not saved): bit-identical isolated vs in-process.
  carry_func     — ``nnsight.carry(x)`` functional form equals the method form.
  not_surfaced   — after an isolated session, the carried var is NOT in the caller frame
                   (it was not saved) while the saved output IS.
  portable       — the same ``.carry()`` code gives identical results in-process and isolated.
  renamed        — renamed model (decoder_blocks): carry handoff isolated == in-process.

Run:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/nnsight-tf/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_session_handoff.py
"""
import sys

import torch

import nnsight
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


def _baseline(model, block):
    with model.trace(P):
        with torch.no_grad():
            d = U(block.output)[:, -1, :].save()
    return d


def _ref(model, src, mid, dst):
    """In-process reference: session, save src.output, patch dst-input with src*1.5,
    read mid downstream. (Works in-process today via the session var-flow.)"""
    with model.session():
        with model.trace(P):
            with torch.no_grad():
                hs = U(src.output).save()
        with model.trace(P) as tracer:
            with torch.no_grad():
                tracer.patch(mid, hs * 1.5)
                d = U(dst.output)[:, -1, :].save()
    return d


def test_saved_handoff(model):
    base = _baseline(model, model.transformer.h[10])
    ref = _ref(model, model.transformer.h[6], model.transformer.h[6], model.transformer.h[10])

    def build():
        with model.session():
            with model.trace(P):
                with torch.no_grad():
                    hs = U(model.transformer.h[6].output).save()
            with model.trace(P) as tracer:
                with torch.no_grad():
                    tracer.patch(model.transformer.h[6], hs * 1.5)
                    d = U(model.transformer.h[10].output)[:, -1, :].save()
        return d

    with isolate_mediators(fast_lane=False, timeout=30):
        got = build()
    ok = _eq(ref, got) and not _eq(got, base)
    print(f"[saved_handoff] isolated == in-process={_eq(ref, got)} "
          f"changed_vs_baseline={not _eq(got, base)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_carry_handoff(model):
    base = _baseline(model, model.transformer.h[10])
    ref = _ref(model, model.transformer.h[6], model.transformer.h[6], model.transformer.h[10])

    def build():
        with model.session():
            with model.trace(P):
                with torch.no_grad():
                    hs = U(model.transformer.h[6].output).carry()       # NOT saved
            with model.trace(P) as tracer:
                with torch.no_grad():
                    tracer.patch(model.transformer.h[6], hs * 1.5)
                    d = U(model.transformer.h[10].output)[:, -1, :].save()
        return d

    with isolate_mediators(fast_lane=False, timeout=30):
        got = build()
    ok = _eq(ref, got) and not _eq(got, base)
    print(f"[carry_handoff] isolated == in-process={_eq(ref, got)} "
          f"changed_vs_baseline={not _eq(got, base)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_carry_func(model):
    ref = _ref(model, model.transformer.h[6], model.transformer.h[6], model.transformer.h[10])

    def build():
        with model.session():
            with model.trace(P):
                with torch.no_grad():
                    hs = nnsight.carry(U(model.transformer.h[6].output))   # functional form
            with model.trace(P) as tracer:
                with torch.no_grad():
                    tracer.patch(model.transformer.h[6], hs * 1.5)
                    d = U(model.transformer.h[10].output)[:, -1, :].save()
        return d

    with isolate_mediators(fast_lane=False, timeout=30):
        got = build()
    ok = _eq(ref, got)
    print(f"[carry_func] nnsight.carry(x) isolated == in-process={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_not_surfaced(model):
    """A carried (non-saved) value must NOT leak to the caller frame; a saved one must."""
    with isolate_mediators(fast_lane=False, timeout=30):
        with model.session():
            with model.trace(P):
                with torch.no_grad():
                    hs = U(model.transformer.h[6].output).carry()
            with model.trace(P) as tracer:
                with torch.no_grad():
                    tracer.patch(model.transformer.h[6], hs * 1.5)
                    down = U(model.transformer.h[10].output)[:, -1, :].save()
        loc = locals()
        down_surfaced = torch.is_tensor(loc.get("down"))
        hs_surfaced = "hs" in loc
    ok = down_surfaced and not hs_surfaced
    print(f"[not_surfaced] saved 'down' surfaced={down_surfaced} "
          f"carried 'hs' surfaced={hs_surfaced} (want True/False)", flush=True)
    return ok


def test_portable(model):
    """The same .carry() code must give identical results in-process and isolated."""
    def build():
        with model.session():
            with model.trace(P):
                with torch.no_grad():
                    hs = U(model.transformer.h[6].output).carry()
            with model.trace(P) as tracer:
                with torch.no_grad():
                    tracer.patch(model.transformer.h[6], hs * 1.5)
                    d = U(model.transformer.h[10].output)[:, -1, :].save()
        return d

    ip = build()
    with isolate_mediators(fast_lane=False, timeout=30):
        iso = build()
    ok = _eq(ip, iso)
    print(f"[portable] .carry() in-process == isolated={ok} (max|Δ|={_delta(ip, iso)})", flush=True)
    return ok


def test_renamed():
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection",
              "transformer.h": "decoder_blocks"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)
    ref = _ref(model, model.decoder_blocks[3], model.decoder_blocks[3], model.decoder_blocks[9])

    def build():
        with model.session():
            with model.trace(P):
                with torch.no_grad():
                    hs = U(model.decoder_blocks[3].output).carry()
            with model.trace(P) as tracer:
                with torch.no_grad():
                    tracer.patch(model.decoder_blocks[3], hs * 1.5)
                    d = U(model.decoder_blocks[9].output)[:, -1, :].save()
        return d

    with isolate_mediators(fast_lane=False, timeout=30):
        got = build()
    ok = _eq(ref, got)
    print(f"[renamed] renamed-model carry handoff isolated == in-process={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "saved_handoff": test_saved_handoff(model),
        "carry_handoff": test_carry_handoff(model),
        "carry_func": test_carry_func(model),
        "not_surfaced": test_not_surfaced(model),
        "portable": test_portable(model),
        "renamed": test_renamed(),
    }
    print("=" * 72, flush=True)
    print(f"ISOLATED SESSION HANDOFF: {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

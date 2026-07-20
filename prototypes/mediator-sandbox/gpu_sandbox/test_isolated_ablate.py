#!/usr/bin/env python3
"""Ablation (zero / mean knockout) on the ISOLATED tier: tracer.ablate replaces a module's
output with a baseline (zeros, or the within-sequence mean) via a REPLACEMENT swap, so the
knockout crosses the isolation boundary and propagates through the host's real forward —
where the hand-written in-place form (``hidden[:] = 0``) silently no-ops (the worker mutates
its delivered clone, no SWAP fires, the host's activation is untouched).

Like tracer.steer/patch (and unlike tracer.unembed), ablation touches no host weights — the
baseline is derived from (or independent of) the delivered activation — so it rides the
existing Events.SWAP with no host round-trip and no isolated/in-process branch: the SAME
method is correct in-process, on the fast lane, and in the isolated worker.

``mode="mean"`` here is the SELF-CONTAINED within-sequence mean (each position → the
per-example mean over the token dimension). Reference-distribution mean ablation (the mean
activation over a dataset, per docs/patterns/ablation.md) is a precomputed value transplanted
via tracer.patch — not this mode.

  zero_single   — zero-ablate one block; forced-isolation == in-process AND downstream !=
                  baseline (the ablation took effect across the boundary).
  mean_single   — mean-ablate one block; forced-isolation == in-process AND != baseline.
  crux          — THE crux: under forced isolation the in-place zero is a no-op (downstream
                  == un-ablated baseline) while tracer.ablate actually ablates (downstream !=
                  baseline) and equals the in-process zero-ablated result.
  tuple_output  — zero-ablate an attention output (a tuple `(tensor, None)`): the whole-tuple
                  replacement branch, forced-isolation == in-process.
  renamed       — renamed model (decoder_blocks): forced-isolation == in-process.
  matches_manual— tracer.ablate (in-process) equals the manual zeros_like / mean-over-seq
                  replacement it names.
  bad_mode      — an unknown mode raises ValueError (no silent wrong-ablation).

Run:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/nnsight-tf/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_ablate.py
"""
import sys

import torch

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


def _baseline_downstream(model, read):
    with model.trace(PROMPT):
        with torch.no_grad():
            o = read.output
            down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
    return down


def test_zero_single(model):
    base = _baseline_downstream(model, model.transformer.h[10])

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.ablate(model.transformer.h[6], "zero")
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got) and not _eq(ref, base)
    print(f"[zero_single] isolated zero-ablate bit-identical={_eq(ref, got)} "
          f"took_effect={not _eq(ref, base)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_mean_single(model):
    base = _baseline_downstream(model, model.transformer.h[10])

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.ablate(model.transformer.h[6], "mean")
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got) and not _eq(ref, base)
    print(f"[mean_single] isolated mean-ablate bit-identical={_eq(ref, got)} "
          f"took_effect={not _eq(ref, base)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_crux(model):
    """The crux: replacement swap crosses the boundary; in-place zero is a silent no-op."""
    base = _baseline_downstream(model, model.transformer.h[10])

    def ablate_downstream():  # tracer.ablate (replacement swap)
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.ablate(model.transformer.h[6], "zero")
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down

    def inplace_downstream():  # hand-written in-place zero
        with model.trace(PROMPT):
            with torch.no_grad():
                out = model.transformer.h[6].output
                hidden = out[0] if isinstance(out, tuple) else out
                hidden[:] = 0
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down

    ablate_ip = ablate_downstream()                      # in-process ablate
    with isolate_mediators(fast_lane=False, timeout=30):
        ablate_iso = ablate_downstream()                 # isolated ablate (replacement swap)
        inplace_iso = inplace_downstream()               # isolated in-place (no-op)

    inplace_is_noop = _eq(inplace_iso, base)             # in-place never crossed the boundary
    ablate_took_effect = not _eq(ablate_iso, base)       # replacement swap changed the host forward
    ablate_correct = _eq(ablate_iso, ablate_ip)          # ... to exactly the in-process result
    ok = inplace_is_noop and ablate_took_effect and ablate_correct
    print(f"[crux] isolated in-place zero is a no-op={inplace_is_noop}; "
          f"isolated ablate took effect={ablate_took_effect}; "
          f"ablate iso==in-process={ablate_correct} (max|Δ|={_delta(ablate_iso, ablate_ip)})", flush=True)
    return ok


def test_tuple_output(model):
    base = _baseline_downstream(model, model.transformer.h[10])

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.ablate(model.transformer.h[6].attn, "zero")
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got) and not _eq(ref, base)
    print(f"[tuple_output] tuple (attn) zero-ablate bit-identical={_eq(ref, got)} "
          f"took_effect={not _eq(ref, base)} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_renamed():
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection",
              "transformer.h": "decoder_blocks"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.ablate(model.decoder_blocks[3], "mean")
                o = model.decoder_blocks[9].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[renamed] renamed-model isolated ablate bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_matches_manual(model):
    # tracer.ablate (in-process) must equal the manual zeros_like / mean-over-seq replacement.
    with model.trace(PROMPT) as tracer:
        with torch.no_grad():
            tracer.ablate(model.transformer.h[6], "zero")
            z_api = (model.transformer.h[10].output[0]
                     if isinstance(model.transformer.h[10].output, tuple)
                     else model.transformer.h[10].output)[:, -1, :].save()
    with model.trace(PROMPT) as tracer:
        with torch.no_grad():
            tracer.ablate(model.transformer.h[6], "mean")
            o = model.transformer.h[10].output
            m_api = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()

    with model.trace(PROMPT):
        with torch.no_grad():
            out = model.transformer.h[6].output
            is_tuple = isinstance(out, tuple)
            hidden = out[0] if is_tuple else out
            zeroed = torch.zeros_like(hidden)
            model.transformer.h[6].output = (zeroed, *out[1:]) if is_tuple else zeroed
            o = model.transformer.h[10].output
            z_manual = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
    with model.trace(PROMPT):
        with torch.no_grad():
            out = model.transformer.h[6].output
            is_tuple = isinstance(out, tuple)
            hidden = out[0] if is_tuple else out
            meaned = hidden.mean(dim=-2, keepdim=True).expand_as(hidden).contiguous()
            model.transformer.h[6].output = (meaned, *out[1:]) if is_tuple else meaned
            o = model.transformer.h[10].output
            m_manual = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()

    ok = _eq(z_api, z_manual) and _eq(m_api, m_manual)
    print(f"[matches_manual] zero == manual zeros_like={_eq(z_api, z_manual)} "
          f"(max|Δ|={_delta(z_api, z_manual)}); mean == manual mean-over-seq={_eq(m_api, m_manual)} "
          f"(max|Δ|={_delta(m_api, m_manual)})", flush=True)
    return ok


def test_bad_mode(model):
    raised = False
    try:
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.ablate(model.transformer.h[6], "median")
                model.transformer.h[10].output  # force the body to run
    except ValueError:
        raised = True
    except Exception as e:  # any other surfaced error is not the contract
        print(f"[bad_mode] wrong exception type: {type(e).__name__}: {e}", flush=True)
    print(f"[bad_mode] unknown mode raises ValueError={raised}", flush=True)
    return raised


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "zero_single": test_zero_single(model),
        "mean_single": test_mean_single(model),
        "crux": test_crux(model),
        "tuple_output": test_tuple_output(model),
        "renamed": test_renamed(),
        "matches_manual": test_matches_manual(model),
        "bad_mode": test_bad_mode(model),
    }
    print("=" * 72, flush=True)
    print(f"ISOLATED ABLATE: {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

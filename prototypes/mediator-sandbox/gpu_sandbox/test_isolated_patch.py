#!/usr/bin/env python3
"""Activation patching (transplant) on the ISOLATED tier: tracer.patch replaces a module's
output with a precomputed value via a REPLACEMENT swap, so the transplant crosses the
isolation boundary and propagates through the host's real forward — where the hand-written
in-place form (``hidden[:] = value``) silently no-ops (the worker mutates its delivered
clone, no SWAP fires, the host's activation is untouched).

Like tracer.steer (and unlike tracer.unembed), patching touches no host weights — only the
delivered activation is replaced — so it rides the existing Events.SWAP with no host
round-trip and no isolated/in-process branch: the SAME method is correct in-process, on the
fast lane, and in the isolated worker. The patch VALUE is precomputed outside the trace (on
CPU) — it crosses the boundary as a transplanted value, mirroring real usage where the
clean/source activation is captured in a prior run.

  single        — patch one block (single-tensor output); forced-isolation downstream
                  residual == in-process, bit-identical: the replacement swap crossed the
                  boundary AND propagated through later layers.
  transplant    — THE crux: under forced isolation the in-place form is a no-op (downstream
                  == unpatched baseline) while tracer.patch actually transplants (downstream
                  != baseline) and equals the in-process patched result.
  tuple_output  — patch an attention output (a tuple `(tensor, None)`): the whole-tuple
                  replacement branch, forced-isolation == in-process.
  multi         — patch three blocks (forward order), forced-isolation == in-process.
  renamed       — renamed model (decoder_blocks): forced-isolation == in-process (no names
                  hardcoded; the wire path is the real path).
  matches_manual— tracer.patch (in-process) equals the manual untuple+replacement (with the
                  same dtype/device cast) it names.

Run:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/nnsight-tf/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_patch.py
"""
import sys

import torch

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


def _patch_value(envoy, model):
    """A full-shape replacement activation, precomputed OUTSIDE the trace (on CPU) so the
    isolated worker transplants a value it never had to compute. Derived from the site's own
    baseline residual (so shapes match by construction) and clearly perturbed (scale+shift)
    so the transplant provably changes the downstream forward."""
    with model.trace(PROMPT):
        with torch.no_grad():
            o = envoy.output
            r = ((o[0] if isinstance(o, tuple) else o) * 1.5 + 0.1).save()
    return r.detach().cpu()


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
    value = _patch_value(model.transformer.h[6], model)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.patch(model.transformer.h[6], value)
                o = model.transformer.h[10].output                 # downstream residual
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[single] isolated patch downstream bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_transplant(model):
    """The crux: replacement swap crosses the boundary; in-place is a silent no-op."""
    value = _patch_value(model.transformer.h[6], model)

    def read_downstream():  # unpatched baseline
        with model.trace(PROMPT):
            with torch.no_grad():
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down

    def patch_downstream():  # tracer.patch (replacement swap)
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.patch(model.transformer.h[6], value)
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down

    def inplace_downstream():  # hand-written in-place transplant
        with model.trace(PROMPT):
            with torch.no_grad():
                out = model.transformer.h[6].output
                hidden = out[0] if isinstance(out, tuple) else out
                hidden[:] = value.to(dtype=hidden.dtype, device=hidden.device)
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down

    base = read_downstream()
    patch_ip = patch_downstream()                        # in-process patch
    with isolate_mediators(fast_lane=False, timeout=30):
        patch_iso = patch_downstream()                   # isolated patch (replacement swap)
        inplace_iso = inplace_downstream()               # isolated in-place (no-op)

    inplace_is_noop = _eq(inplace_iso, base)             # in-place never crossed the boundary
    patch_took_effect = not _eq(patch_iso, base)         # replacement swap changed the host forward
    patch_correct = _eq(patch_iso, patch_ip)             # ... to exactly the in-process result
    ok = inplace_is_noop and patch_took_effect and patch_correct
    print(f"[transplant] isolated in-place is a no-op={inplace_is_noop}; "
          f"isolated patch took effect={patch_took_effect}; "
          f"patch iso==in-process={patch_correct} (max|Δ|={_delta(patch_iso, patch_ip)})", flush=True)
    return ok


def test_tuple_output(model):
    # An attention output is a tuple (tensor, None) — exercises the whole-tuple replacement
    # branch (transplanted element [0], None tail carried through).
    value = _patch_value(model.transformer.h[6].attn, model)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.patch(model.transformer.h[6].attn, value)
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[tuple_output] tuple (attn) patch bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_multi(model):
    vals = {i: _patch_value(model.transformer.h[i], model) for i in [2, 5, 8]}

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                for i in [2, 5, 8]:                       # forward order
                    tracer.patch(model.transformer.h[i], vals[i])
                o = model.transformer.h[11].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[multi] 3-block isolated patch bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_renamed():
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection",
              "transformer.h": "decoder_blocks"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)
    value = _patch_value(model.decoder_blocks[3], model)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.patch(model.decoder_blocks[3], value)
                o = model.decoder_blocks[9].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[renamed] renamed-model isolated patch bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_matches_manual(model):
    # tracer.patch (in-process) must equal the manual untuple + replacement (with cast) it names.
    value = _patch_value(model.transformer.h[6], model)
    with model.trace(PROMPT) as tracer:
        with torch.no_grad():
            tracer.patch(model.transformer.h[6], value)
            o = model.transformer.h[10].output
            via_api = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
    with model.trace(PROMPT):
        with torch.no_grad():
            out = model.transformer.h[6].output
            is_tuple = isinstance(out, tuple)
            hidden = out[0] if is_tuple else out
            v = value.to(dtype=hidden.dtype, device=hidden.device)
            model.transformer.h[6].output = (v, *out[1:]) if is_tuple else v
            o = model.transformer.h[10].output
            via_manual = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
    ok = _eq(via_api, via_manual)
    print(f"[matches_manual] tracer.patch == manual replacement bit-identical={ok} "
          f"(max|Δ|={_delta(via_api, via_manual)})", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "single": test_single(model),
        "transplant": test_transplant(model),
        "tuple_output": test_tuple_output(model),
        "multi": test_multi(model),
        "renamed": test_renamed(),
        "matches_manual": test_matches_manual(model),
    }
    print("=" * 72, flush=True)
    print(f"ISOLATED PATCH: {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

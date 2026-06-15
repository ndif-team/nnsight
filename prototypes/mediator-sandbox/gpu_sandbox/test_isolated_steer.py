#!/usr/bin/env python3
"""Activation steering on the ISOLATED tier: tracer.steer injects a direction into a
module's output via a REPLACEMENT swap, so the steering crosses the isolation boundary and
propagates through the host's real forward — where the hand-written in-place form
(``hidden[:] = …``) silently no-ops (the worker mutates its delivered clone, no SWAP fires,
the host's activation is untouched).

Unlike tracer.unembed, steering touches no host weights — only the delivered activation —
so it needs no host round-trip: the eproperty setter routes the SWAP on either tier and the
SAME method is correct in-process, on the fast lane, and in the isolated worker. The
steering DIRECTION is precomputed outside the trace (a host-weight read inside a forced-
isolation trace would hit the weightless worker — the unembed problem); that mirrors real
usage, where steering vectors are precomputed.

  single        — steer one block (single-tensor output), forced-isolation downstream
                  residual == in-process, bit-identical: the replacement swap crossed the
                  boundary AND propagated through later layers.
  replacement   — THE crux: under forced isolation the in-place form is a no-op (downstream
                  == unsteered baseline) while tracer.steer actually steers (downstream !=
                  baseline) and equals the in-process steered result.
  tuple_output  — steer an attention output (a tuple `(tensor, None)`): the whole-tuple
                  replacement branch, forced-isolation == in-process.
  multi         — steer three blocks (forward order), forced-isolation == in-process.
  renamed       — renamed model (decoder_blocks): forced-isolation == in-process (no names
                  hardcoded; the wire path is the real path).
  matches_manual— tracer.steer (in-process) equals the manual untuple+replacement it names.

Run:
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_steer.py
"""
import sys

import torch
import torch.nn.functional as F

from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"
ALPHA = 8.0


def _direction(head, token_id=5000):
    """A precomputed unit steering vector, on CPU and detached — captured into the trace
    body so the isolated worker never reads a host weight (it has none)."""
    return F.normalize(head.weight[token_id].float(), dim=0).detach().cpu()


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
    direction = _direction(model.lm_head)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.steer(model.transformer.h[6], direction, ALPHA)
                o = model.transformer.h[10].output                 # downstream residual
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[single] isolated steer downstream bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_replacement(model):
    """The crux: replacement swap crosses the boundary; in-place is a silent no-op."""
    direction = _direction(model.lm_head)

    def read_downstream():  # unsteered baseline
        with model.trace(PROMPT):
            with torch.no_grad():
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down

    def steer_downstream():  # tracer.steer (replacement swap)
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.steer(model.transformer.h[6], direction, ALPHA)
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down

    def inplace_downstream():  # hand-written in-place steering
        with model.trace(PROMPT):
            with torch.no_grad():
                out = model.transformer.h[6].output
                hidden = out[0] if isinstance(out, tuple) else out
                hidden[:] = hidden + ALPHA * direction.to(dtype=hidden.dtype, device=hidden.device)
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down

    base = read_downstream()
    steer_ip = steer_downstream()                        # in-process steer
    with isolate_mediators(fast_lane=False, timeout=30):
        steer_iso = steer_downstream()                   # isolated steer (replacement swap)
        inplace_iso = inplace_downstream()               # isolated in-place (no-op)

    inplace_is_noop = _eq(inplace_iso, base)             # in-place never crossed the boundary
    steer_took_effect = not _eq(steer_iso, base)         # replacement swap changed the host forward
    steer_correct = _eq(steer_iso, steer_ip)             # ... to exactly the in-process result
    ok = inplace_is_noop and steer_took_effect and steer_correct
    print(f"[replacement] isolated in-place is a no-op={inplace_is_noop}; "
          f"isolated steer took effect={steer_took_effect}; "
          f"steer iso==in-process={steer_correct} (max|Δ|={_delta(steer_iso, steer_ip)})", flush=True)
    return ok


def test_tuple_output(model):
    # An attention output is a tuple (tensor, None) — exercises the whole-tuple replacement
    # branch (steered element [0], None tail carried through).
    direction = _direction(model.lm_head)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.steer(model.transformer.h[6].attn, direction, ALPHA)
                o = model.transformer.h[10].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[tuple_output] tuple (attn) steer bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_multi(model):
    direction = _direction(model.lm_head)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                for i in [2, 5, 8]:                       # forward order
                    tracer.steer(model.transformer.h[i], direction, ALPHA)
                o = model.transformer.h[11].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[multi] 3-block isolated steer bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_renamed():
    rename = {"transformer.ln_f": "final_norm", "lm_head": "output_projection",
              "transformer.h": "decoder_blocks"}
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True, rename=rename)
    direction = _direction(model.output_projection)

    def build():
        with model.trace(PROMPT) as tracer:
            with torch.no_grad():
                tracer.steer(model.decoder_blocks[3], direction, ALPHA)
                o = model.decoder_blocks[9].output
                down = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
        return down
    ref, got = _both(build)
    ok = _eq(ref, got)
    print(f"[renamed] renamed-model isolated steer bit-identical={ok} (max|Δ|={_delta(ref, got)})", flush=True)
    return ok


def test_matches_manual(model):
    # tracer.steer (in-process) must equal the manual untuple + replacement it names.
    direction = _direction(model.lm_head)
    with model.trace(PROMPT) as tracer:
        with torch.no_grad():
            tracer.steer(model.transformer.h[6], direction, ALPHA)
            o = model.transformer.h[10].output
            via_api = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
    with model.trace(PROMPT):
        with torch.no_grad():
            out = model.transformer.h[6].output
            is_tuple = isinstance(out, tuple)
            hidden = out[0] if is_tuple else out
            steered = hidden + ALPHA * direction.to(dtype=hidden.dtype, device=hidden.device)
            model.transformer.h[6].output = (steered, *out[1:]) if is_tuple else steered
            o = model.transformer.h[10].output
            via_manual = (o[0] if isinstance(o, tuple) else o)[:, -1, :].save()
    ok = _eq(via_api, via_manual)
    print(f"[matches_manual] tracer.steer == manual replacement bit-identical={ok} "
          f"(max|Δ|={_delta(via_api, via_manual)})", flush=True)
    return ok


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)
    results = {
        "single": test_single(model),
        "replacement": test_replacement(model),
        "tuple_output": test_tuple_output(model),
        "multi": test_multi(model),
        "renamed": test_renamed(),
        "matches_manual": test_matches_manual(model),
    }
    print("=" * 72, flush=True)
    print(f"ISOLATED STEER: {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

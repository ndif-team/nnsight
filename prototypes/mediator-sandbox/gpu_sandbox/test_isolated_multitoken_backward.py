#!/usr/bin/env python3
"""Characterize multi-token (generate + iter) backward under isolation.

The prior review CLAIMED per-step retention overwrite ("last step wins") would make
this silently wrong, but the severity probes were inconclusive (the in-process control
constructions themselves errored). This script builds the control FIRST and compares
outcome-for-outcome:

  per_step  — backward inside the iter loop at each step: read ln_f.output, loss =
              lm_head.output.sum() at that step, read hs.grad inside the backward block.
  post_loop — accumulate ln_f.output per step, after the loop build a scalar loss from
              one chosen step's activation and read its .grad in a backward block.

Verdicts per shape:
  both succeed + bit-identical        -> SUPPORTED (record in the doc matrix)
  both succeed + values differ        -> SILENT-WRONG (guard needed)
  control fails                       -> in-process doesn't support it either; isolated
                                         must fail too, with a non-cryptic error
  control succeeds + isolated fails   -> isolation gap (clean error = acceptable,
                                         documented; hang/cryptic = fix)

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_multitoken_backward.py
"""
import sys
import traceback

import torch

import nnsight
from nnsight import LanguageModel
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"
N = 3  # max_new_tokens; backward exercised at steps 0 and 1


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _outcome(fn):
    """Run fn; return ("ok", value) or ("err", "<Type>: <msg>")."""
    try:
        return "ok", fn()
    except Exception as e:  # noqa: BLE001
        return "err", f"{type(e).__name__}: {e}"


def _per_step(model, iso):
    """Backward inside the iter loop: per-step grad of ln_f.output w.r.t. that
    step's sum-of-logits loss."""

    def body():
        ctx = isolate_mediators(fast_lane=False, timeout=60) if iso else _null()
        with ctx:
            with model.generate(PROMPT, max_new_tokens=N) as t:
                grads = []
                for step in t.iter[:2]:
                    hs = model.transformer.ln_f.output
                    loss = model.lm_head.output.sum()
                    with loss.backward():
                        grads.append(hs.grad)
                nnsight.save(grads)
        return grads

    return _outcome(body)


def _post_loop(model, iso):
    """Backward after the iter loop on one chosen step's activation: loss derived
    from the step-1 activation itself; .grad read in the backward block."""

    def body():
        ctx = isolate_mediators(fast_lane=False, timeout=60) if iso else _null()
        with ctx:
            with model.generate(PROMPT, max_new_tokens=N) as t:
                hs = []
                for step in t.iter[:2]:
                    hs.append(model.transformer.ln_f.output)
                loss = (hs[1].float() ** 2).sum()
                with loss.backward():
                    g = hs[1].grad.save()
        return g

    return _outcome(body)


def _compare(name, ref, got):
    """ref/got are (status, value) outcomes. Print verdict, return pass/fail."""
    rs, rv = ref
    gs, gv = got

    if rs == "err" and gs == "err":
        # CHARACTERIZED (2026-06-10): generate() runs the forward without gradient
        # tracking, so multi-token backward fails IN-PROCESS at the first .grad read
        # ("cannot register a hook on a tensor that doesn't require gradient").
        # No silent-wrong is possible — there is no graph at all. The isolated path
        # must fail with the message naming that real cause (not "off the backward
        # path", which blames the wrong thing).
        print(f"[{name}] control errors -> not supported in-process either.")
        print(f"[{name}]   in-proc : {' '.join(str(rv).split())[:160]}")
        print(f"[{name}]   isolated: {' '.join(str(gv).split())[:160]}")
        clear = "without gradient tracking" in str(gv)
        verdict = (
            "CLEAN-FAIL (parity, cause named)"
            if clear
            else "CRYPTIC-FAIL (isolated error must name the grad-less forward)"
        )
        print(f"[{name}] verdict: {verdict}", flush=True)
        return clear

    if rs == "ok" and gs == "err":
        print(f"[{name}] control OK but isolated errors: {str(gv)[:200]}")
        print(f"[{name}] verdict: ISOLATION GAP (clean error; document or fix)", flush=True)
        return False

    if rs == "err" and gs == "ok":
        print(f"[{name}] isolated SUCCEEDS where in-process errors ({str(rv)[:120]})")
        print(f"[{name}] verdict: SEMANTIC DIVERGENCE (isolated more permissive — investigate)", flush=True)
        return False

    # Both ok: compare values (tensor or list of tensors).
    rl = rv if isinstance(rv, list) else [rv]
    gl = gv if isinstance(gv, list) else [gv]
    if len(rl) != len(gl) or not all(torch.is_tensor(a) and torch.is_tensor(b) for a, b in zip(rl, gl)):
        print(f"[{name}] BOTH OK but shapes of result differ: ref={rl} got={gl}")
        print(f"[{name}] verdict: SILENT-WRONG (structure mismatch)", flush=True)
        return False
    same = all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(rl, gl))
    deltas = [
        (a - b).abs().max().item() if a.shape == b.shape else float("nan")
        for a, b in zip(rl, gl)
    ]
    shapes = [tuple(a.shape) for a in rl]
    verdict = "SUPPORTED (bit-identical)" if same else "SILENT-WRONG (values differ)"
    print(f"[{name}] both OK: shapes={shapes} max|Δ| per step={deltas}")
    print(f"[{name}] verdict: {verdict}", flush=True)
    return same


def main():
    assert torch.cuda.is_available()
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)

    results = {}
    for name, fn in (("per_step", _per_step), ("post_loop", _post_loop)):
        print(f"--- {name}: in-process control ---", flush=True)
        ref = fn(model, iso=False)
        print(f"--- {name}: isolated ---", flush=True)
        got = fn(model, iso=True)
        results[name] = _compare(name, ref, got)

    print("=" * 72, flush=True)
    print(f"MULTI-TOKEN BACKWARD CHARACTERIZATION: {results}", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)

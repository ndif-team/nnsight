#!/usr/bin/env python3
"""Unit tests for the fast-lane safety classifier (GPU-free, no model).

Confirms the detect-and-confirm gate on the ACTUAL interp-workload shapes (logit lens,
steering, activation patching, attribution) plus footgun payloads, and — per the testing
rules — on RENAMED module structures so nothing is keyed to GPT-2 naming. The classifier
walks the effective code (closures resolved through globals/closure cells), so these tests
use the same closure-wrapped shape the real harness uses (a build()/capture() lambda
calling helper functions).

Run:
  PYTHONPATH=src /disk/u/zikai/anaconda3/envs/hf-serve/bin/python -u \
    prototypes/mediator-sandbox/gpu_sandbox/test_fastlane_classifier.py
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnsight.intervention.fastlane import FAST, ISOLATE, REJECT, classify_callable


# --- stand-ins for the host objects a trace body closes over -------------------------
# Real nn.Modules so the classifier's host-object detection (calling them / reading
# .weight) exercises the genuine path, not a mock.
class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.output = torch.zeros(1, 3, d)  # stands in for the eproperty read target

    def forward(self, x):
        return self.lin(x)


def _make_model(norm_name="ln_f", head_name="lm_head", blocks_name="h", d=8, n=4):
    """A tiny model with CONFIGURABLE attribute names — the renamed-structure test tool."""
    m = nn.Module()
    setattr(m, blocks_name, nn.ModuleList([_Block(d) for _ in range(n)]))
    setattr(m, norm_name, nn.LayerNorm(d))
    setattr(m, head_name, nn.Linear(d, 50, bias=False))
    return m


# --- helper functions a "trace body" calls (the closure chain) ------------------------
def _untuple(x):
    return x[0] if isinstance(x, tuple) else x


def _lens_proxy(blocks, norm, head, *, layers):
    rows = []
    with torch.no_grad():
        for i in (range(len(blocks)) if layers == "all" else layers):
            normed = norm(_untuple(blocks[i].output))
            logits = F.linear(normed, head.weight)   # host-weight read — fast-lane-only
            rows.append(logits[:, -1, :])
    return torch.stack(rows, dim=0)


def _steer_inplace(blocks, head, *, layer, token_id, alpha):
    with torch.no_grad():
        direction = F.normalize(head.weight[token_id].float(), dim=0)
        out = blocks[layer].output
        hidden = out[0] if isinstance(out, tuple) else out
        hidden[:] = hidden + alpha * direction   # in-place write (in_place flag)
        return hidden


# =====================================================================================
CASES = []


def case(expect):
    def deco(fn):
        CASES.append((fn.__name__, fn, expect))
        return fn
    return deco


# ---- SAFE interp workloads (must classify FAST) -------------------------------------
@case(FAST)
def logit_lens_gpt2_style():
    m = _make_model()
    return lambda: _lens_proxy(m.h, m.ln_f, m.lm_head, layers="all").save()


@case(FAST)
def logit_lens_renamed_structure():
    # non-GPT-2 names: decoder_blocks / final_norm / output_projection
    m = _make_model(norm_name="final_norm", head_name="output_projection",
                    blocks_name="decoder_blocks")
    return lambda: _lens_proxy(m.decoder_blocks, m.final_norm, m.output_projection,
                               layers=[0, 2]).save()


@case(FAST)
def steering_inplace_is_fast():
    # in-place steering is the fast lane's correctness win (silent no-op under isolation)
    m = _make_model()
    return lambda: _steer_inplace(m.h, m.lm_head, layer=1, token_id=5, alpha=6.0).save()


@case(FAST)
def boundary_replacement_write():
    m = _make_model()

    def body():
        out = m.h[1].output
        new = out * 2.0
        m.h[2].output = new          # nnsight boundary write (SWAP), allowed
        return m.ln_f(new)[:, -1, :]
    return body


@case(FAST)
def backward_attribution_shape():
    m = _make_model()

    def body():
        a = m.h[2].output
        a.requires_grad_(True)
        normed = m.ln_f(a)
        metric = F.linear(normed, m.lm_head.weight).sum()
        with metric.backward():
            g = a.grad.save()
        return g
    return body


# ---- footguns that must NOT reach the fast lane -------------------------------------
@case(ISOLATE)
def imports_isolate():
    def body():
        import os
        return os.getpid()
    return body


@case(ISOLATE)
def while_loop_isolate():
    def body():
        x = 0
        while x < 10:
            x = x + 1
        return x
    return body


@case(ISOLATE)
def unresolved_global_call_isolate():
    def body():
        return some_undefined_helper(3)   # noqa: F821 — unknown authority
    return body


@case(ISOLATE)
def open_file_isolate():
    def body():
        return open("/etc/passwd").read()
    return body


@case(REJECT)
def introspection_subclasses_reject():
    def body():
        return ().__class__.__bases__
    return body


@case(REJECT)
def getattr_escape_reject():
    def body():
        return getattr(torch, "save")
    return body


@case(REJECT)
def dunder_subscript_reject():
    def body():
        d = {}
        return d["__builtins__"]
    return body


@case(ISOLATE)
def host_attr_write_isolate():
    m = _make_model()

    def body():
        m.ln_f.eps = 1.0     # mutating host state visible to siblings
        return m.ln_f.weight
    return body


# ---- flag detection ----------------------------------------------------------------
def main():
    results = {}
    for name, factory, expect in CASES:
        fn = factory()
        v = classify_callable(fn)
        ok = v.tier == expect
        results[name] = ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {v.tier} expected {expect} "
              f"(diff={v.differentiate} inplace={v.in_place} weights={v.touches_host_weights}) "
              f"-- {v.reason}", flush=True)

    # flag assertions on specific cases
    flag_checks = {
        "backward sets differentiate": classify_callable(backward_attribution_shape()).differentiate is True,
        "steering sets in_place": classify_callable(steering_inplace_is_fast()).in_place is True,
        "lens reads host weights": classify_callable(logit_lens_gpt2_style()).touches_host_weights is True,
        "replacement write is not in_place": classify_callable(boundary_replacement_write()).in_place is False,
    }
    for k, v in flag_checks.items():
        results[k] = v
        print(f"[{'PASS' if v else 'FAIL'}] {k}", flush=True)

    print("=" * 72, flush=True)
    npass = sum(results.values())
    print(f"FAST-LANE CLASSIFIER: {npass}/{len(results)} passed", flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

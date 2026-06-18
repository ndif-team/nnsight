#!/usr/bin/env python3
"""Triton-kernel model under isolation — the headline motivation for this backend.

A model whose forward executes a Triton JIT kernel cannot be served under NDIF's
in-process import-whitelist sandbox: Triton's first-use compilation needs subprocess
(ptxas), tempfile, open() and importlib — exactly what the whitelist denies — and the
whitelist necessarily wraps the forward pass too. This backend moves the forward to the
TRUSTED host (unrestricted, so Triton compiles normally) and contains only the UNTRUSTED
intervention in the worker. These tests pin that property:

  host_compiles   — an isolated trace with ``lockdown=True`` through a Triton-kernel model
                    is bit-identical to in-process. The worker is fully seccomp'd, so the
                    kernel MUST have compiled+run on the host => the host/worker split lets
                    Triton models run under isolation. (A fresh TRITON_CACHE_DIR is shown
                    to populate during the isolated run as corroboration.)
  user_contained  — a Triton kernel invoked from WITHIN the intervention (worker-side) under
                    lockdown fails cleanly (import wall and/or compile wall), demonstrating
                    that the capability preserved for the model is still denied to user code.
                    Uses ``preimport=("triton", "triton.language")`` to get the worker past
                    the import wall so the *compile* wall is what's exercised.

Run (needs a GPU + a Triton install):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_triton_model.py
"""
import os
import sys
import tempfile

# Point Triton at a fresh cache BEFORE importing it, so a populated cache after the
# isolated run is evidence the host compiled during that run (not a pre-warmed hit).
_CACHE_DIR = tempfile.mkdtemp(prefix="nnsight_triton_cache_")
os.environ["TRITON_CACHE_DIR"] = _CACHE_DIR

import torch
import torch.nn as nn

import nnsight
from nnsight import NNsight
from nnsight.intervention.isolation import isolate_mediators

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _scale_kernel(x_ptr, out_ptr, scale, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * scale, mask=mask)

    def triton_scale(x, scale):
        """Elementwise x*scale via a Triton kernel (forces a Triton JIT at first use)."""
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
        _scale_kernel[grid](x, out, scale, n, BLOCK=1024)
        return out


class TritonMLP(nn.Module):
    """fc1 -> (Triton kernel) -> fc2. The Triton kernel lives ON the forward path, so a
    trace of this model only succeeds if that kernel compiled+ran wherever the forward
    executed (the host, under isolation)."""

    def __init__(self, d=64):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        h = self.fc1(x)
        h = triton_scale(h, 2.0)
        return self.fc2(h)


def _build():
    torch.manual_seed(0)
    net = TritonMLP().cuda().eval()
    return NNsight(net), torch.randn(1, 8, 64).cuda()


def test_host_compiles():
    """Isolated + lockdown trace through a Triton-kernel model == in-process."""
    model, x = _build()

    # Isolated FIRST, with a cold Triton cache and the worker fully locked down.
    with isolate_mediators(lockdown=True):
        with model.trace(x):
            got = model.fc2.output.save()

    cache_populated = os.path.isdir(_CACHE_DIR) and any(os.scandir(_CACHE_DIR))

    # In-process reference (cache now warm; comparison is what matters).
    with model.trace(x):
        ref = model.fc2.output.save()

    ok = torch.equal(ref, got)
    print(
        f"[host_compiles] isolated(lockdown) Triton model == in-process: {ok} "
        f"(max|Δ|={(ref - got).abs().max().item():.2e}) | host triton cache populated "
        f"during isolated run: {cache_populated}"
    )
    # Bit-identity is the load-bearing assertion; cache population is corroboration only
    # (cache layout/location is version-dependent), so it does not gate the result.
    return ok


def test_user_contained():
    """A Triton kernel called from inside the intervention (worker) is blocked under
    lockdown — the model keeps Triton, untrusted user code does not."""
    model, x = _build()
    raised = None
    try:
        # preimport gets the worker past the import wall; the compile wall (ptxas execve +
        # cache open, both seccomp-blocked) is then what must stop it.
        with isolate_mediators(lockdown=True, preimport=("triton", "triton.language")):
            with model.trace(x):
                act = model.fc1.output          # real tensor in the worker
                # Worker-side Triton use: the JIT (ptxas execve + cache open) must be
                # blocked under lockdown. This call is what should raise.
                out = triton_scale(act, 3.0)
                nnsight.save(out)
    except Exception as e:  # noqa: BLE001
        raised = e
    ok = raised is not None
    print(
        f"[user_contained] worker-side Triton blocked under lockdown: {ok} "
        f"(raised={type(raised).__name__ if raised else None})"
    )
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        sys.exit(0)
    if not _HAVE_TRITON:
        print("SKIP: triton not installed")
        sys.exit(0)

    results = {
        "host_compiles": test_host_compiles(),
        "user_contained": test_user_contained(),
    }
    ok = all(results.values())
    print("=" * 72)
    print(f"ISOLATED TRITON MODEL: {'PASS' if ok else 'FAIL'} — {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

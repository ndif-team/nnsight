#!/usr/bin/env python3
"""CudaIpcChannel buffer codec tests (in-process, no spawn).

Tests the new GPU-bounce-buffer tensor pack/unpack that the CudaIpcChannel rides on:

  1. round-trip   — a nested (tensor, dict, tuple, scalar, None) value packs into a
                    shared GPU buffer (offset table + non-tensor skeleton) and unpacks
                    bit-identically.
  2. aliasing     — unpacked tensors are CLONED out of the buffer, so overwriting the
                    buffer afterwards does NOT corrupt them (the one-event-in-flight
                    clone-on-receive correctness rule, design §2).

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_cuda_channel_codec.py
"""
import sys

import torch

from nnsight.intervention.transport import pack_cuda, unpack_cuda

DEV = "cuda"
ARENA = 64 << 20


def _equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b or (a is None and b is None)


def test_roundtrip():
    buf = torch.empty(ARENA, dtype=torch.uint8, device=DEV)
    value = (
        torch.randn(2, 5, 7, device=DEV),
        {"k": torch.arange(3, device=DEV), "scale": 2.5},
        torch.randn(1, 16, 768, device=DEV, dtype=torch.bfloat16),  # block-output-ish
        "meta",
        None,
    )
    skel, table = pack_cuda(value, buf)
    got = unpack_cuda(skel, table, buf)
    ok = _equal(value, got)
    print(f"[1 roundtrip] nested (tensor/dict/tuple/bf16/scalar/None) bit-identical: {ok}")
    return ok


def test_aliasing_clone_on_receive():
    buf = torch.empty(ARENA, dtype=torch.uint8, device=DEV)
    t = torch.arange(100, device=DEV, dtype=torch.float32)
    skel, table = pack_cuda((t,), buf)
    got = unpack_cuda(skel, table, buf)
    before = got[0].clone()
    # Simulate the next event reusing the single buffer:
    buf.fill_(0)
    survived = torch.equal(got[0], before) and torch.equal(got[0], t)
    print(f"[2 aliasing]  unpacked tensor survives buffer overwrite (cloned): {survived}")
    return survived


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    results = {
        "roundtrip": test_roundtrip(),
        "aliasing": test_aliasing_clone_on_receive(),
    }
    ok = all(results.values())
    print("=" * 72)
    print(f"CHANNEL CODEC: {'PASS' if ok else 'FAIL'} — {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

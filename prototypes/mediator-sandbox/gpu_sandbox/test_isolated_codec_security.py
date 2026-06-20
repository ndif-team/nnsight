#!/usr/bin/env python3
"""Worker->host frame codec — fidelity + security (closed value algebra, no pickle VM).

The isolated worker runs UNTRUSTED user code, so the host must never run pickle's VM on
its frames (a ``__reduce__`` gadget would execute on the trusted host = RCE). Instead the
boundary transmits a CLOSED VALUE ALGEBRA (``transport._codec_dumps`` / ``_codec_loads``):
None/bool/int/float/str/bytes, list/tuple/dict/set, torch dtype/device, and out-of-band
ARRAY leaves (torch tensors AND numpy arrays ride the GPU buffer / safetensors). There is no
opcode that can call a function, so decoding is pure data assembly.

  fidelity   - VALUE / SWAP / END / EXCEPTION / cross_invoker-push, a tracer.cache()-style
               spec carrying torch.dtype + torch.device, AND a numpy array all round-trip
               exactly (numpy re-materializes as ndarray).
  security   - a value OUTSIDE the algebra (a __reduce__ gadget, a custom class) is rejected
               at ENCODE, in the worker, before any pickle/__reduce__ runs — never an
               encode-ok / decode-refuse split.
  robustness - malformed / oversized / unknown-tag bytes raise BoundaryDecodeError host-side
               (no crash, no hang), not a silent or VM-level failure.

Needs torch + numpy; **CPU is enough — no CUDA required** (the bounce buffer is just bytes).
Run:
  PYTHONPATH=src /disk/u/zikai/anaconda3/envs/nnsight-tf/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_codec_security.py
"""
import os
import sys

import numpy as np
import torch

from nnsight.intervention import transport as T
from nnsight.intervention.interleaver import Events
from nnsight.intervention.transport import BoundaryDecodeError, BoundaryValueError


def _roundtrip(event, data, push=None, nbytes=1 << 20):
    """Encode a worker->host frame and decode it host-side, via a CPU 'bounce buffer'."""
    buf = torch.empty(nbytes, dtype=torch.uint8)  # CPU stand-in for the GPU buffer
    frame, _had = T._encode_worker_frame(event, data, push, buf)
    return T._decode_worker_frame(frame, buf)


def test_fidelity():
    ok = True

    # VALUE: the requester string.
    ev, out, _ = _roundtrip(Events.VALUE, "transformer.h.6.output.i0")
    ok &= ev is Events.VALUE and out == "transformer.h.6.output.i0"

    # SWAP: (requester, value) carrying a tensor + a None tail (the attn-output shape).
    t = torch.randn(2, 3)
    ev, out, _ = _roundtrip(Events.SWAP, ("h.6.output", (t, None)))
    ok &= ev is Events.SWAP and out[0] == "h.6.output"
    ok &= torch.equal(out[1][0], t) and out[1][1] is None

    # END: saved dict — tensor + scalars + bytes + set + nested list/tuple.
    saved = {
        "x": torch.arange(6).float(),
        "n": 3, "f": 1.5, "b": b"hi", "set": {1, 2, 3},
        "lst": [1, (2, 3)],
        "tup": (torch.ones(2), None),
    }
    ev, out, _ = _roundtrip(Events.END, saved)
    ok &= ev is Events.END
    ok &= torch.equal(out["x"], saved["x"]) and out["n"] == 3 and out["f"] == 1.5
    ok &= out["b"] == b"hi" and out["set"] == {1, 2, 3}
    ok &= out["lst"] == [1, (2, 3)] and type(out["lst"][1]) is tuple
    ok &= type(out["tup"]) is tuple and torch.equal(out["tup"][0], torch.ones(2))
    ok &= out["tup"][1] is None

    # CACHE: spec carrying torch.dtype + torch.device (the previously-missed case).
    spec = (12345, ["transformer.h.0"], torch.device("cpu"), torch.float16,
            True, True, False, {}, {})
    ev, out, _ = _roundtrip(Events.CACHE, spec)
    ok &= ev is Events.CACHE
    ok &= out[2] == torch.device("cpu") and out[3] is torch.float16
    ok &= out[0] == 12345 and out[1] == ["transformer.h.0"]

    # NUMPY: a saved ndarray crosses out-of-band (the Array leaf) and comes back ndarray.
    # (A numpy *scalar* is not an ndarray, so it's not in the algebra — save a python value.)
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    ev, out, _ = _roundtrip(Events.END, {"np": arr})
    ok &= isinstance(out["np"], np.ndarray) and out["np"].dtype == np.float32
    ok &= np.array_equal(out["np"], arr)

    # EXCEPTION: rebuilt host-side from (type-name, message); no object crosses.
    ev, out, _ = _roundtrip(Events.EXCEPTION, ValueError("boom"))
    ok &= ev is Events.EXCEPTION and isinstance(out, ValueError) and "boom" in str(out)

    # cross_invoker push: CPU tensors + scalars ride safetensors, not the value codec.
    push = {"shared": torch.randn(4), "k": 7}
    ev, _out, pout = _roundtrip(Events.END, {"y": torch.zeros(1)}, push=push)
    ok &= pout is not None and torch.equal(pout["shared"], push["shared"]) and pout["k"] == 7

    print(f"[fidelity] VALUE/SWAP/END/CACHE(dtype,device)/numpy/EXCEPTION/push exact: {ok}")
    return ok


def test_security():
    probe = "/tmp/nnsight_codec_pwn_probe"
    if os.path.exists(probe):
        os.remove(probe)

    class Bomb:  # a classic __reduce__ gadget
        def __reduce__(self):
            return (os.system, (f"echo pwned > {probe}",))

    class Plain:  # a harmless but non-algebra custom object
        pass

    buf = torch.empty(1 << 16, dtype=torch.uint8)

    # The gadget is OUTSIDE the value algebra, so the codec refuses it at ENCODE, in the
    # worker — __reduce__ is never even consulted, nothing reaches the host.
    gadget_rejected = False
    try:
        T._encode_worker_frame(Events.END, {"evil": Bomb()}, None, buf)
    except BoundaryValueError:
        gadget_rejected = True
    no_pwn = not os.path.exists(probe)
    if os.path.exists(probe):
        os.remove(probe)

    # A harmless custom object is refused the same way — the algebra is closed, not a denylist.
    plain_rejected = False
    try:
        T._encode_worker_frame(Events.END, {"obj": Plain()}, None, buf)
    except BoundaryValueError:
        plain_rejected = True

    ok = gadget_rejected and no_pwn and plain_rejected
    print(f"[security] gadget rejected at encode (never ran): {gadget_rejected and no_pwn} | "
          f"custom object rejected at encode: {plain_rejected}")
    return ok


def test_robustness():
    # The host decodes UNTRUSTED bytes: malformed / oversized / unknown-tag must raise a
    # clean BoundaryDecodeError, never crash or hang.
    buf = torch.empty(1 << 16, dtype=torch.uint8)
    from nnsight.intervention.transport import _codec_loads, _MAX_CODEC_BYTES

    cases = {
        "truncated": bytes([T._T_LIST, 0xFF, 0xFF, 0x01]),     # claims items, no data
        "unknown_tag": bytes([200]),                            # tag not in the algebra
        "trailing": bytes([T._T_NONE, T._T_NONE]),              # two values, one expected
        "bad_varint": bytes([T._T_STR] + [0x80] * 12),          # runaway length varint
        "oversized": b"\x00" * (_MAX_CODEC_BYTES + 1),          # past the size cap
    }
    ok = True
    for name, payload in cases.items():
        raised = False
        try:
            _codec_loads(payload)
        except BoundaryDecodeError:
            raised = True
        except Exception as e:  # any other exception = not a clean rejection
            print(f"  [{name}] WRONG exception {type(e).__name__}")
        ok &= raised
    print(f"[robustness] malformed/oversized/unknown-tag all raise BoundaryDecodeError: {ok}")
    return ok


def main():
    results = {
        "fidelity": test_fidelity(),
        "security": test_security(),
        "robustness": test_robustness(),
    }
    ok = all(results.values())
    print("=" * 72)
    print(f"ISOLATED CODEC (value algebra): {'PASS' if ok else 'FAIL'} — {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

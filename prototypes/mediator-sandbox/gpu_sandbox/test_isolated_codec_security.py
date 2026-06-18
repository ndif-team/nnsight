#!/usr/bin/env python3
"""Worker->host frame codec — security + fidelity (restricted unpickler).

The isolated worker runs UNTRUSTED user code; the host must not *plain* ``pickle.loads``
its frames (a ``__reduce__`` gadget would execute on the trusted host = RCE). Worker->host
frames are tensor-free (tensors ride the GPU buffer / safetensors) and the rest is decoded
with ``transport._RestrictedUnpickler`` (``find_class`` allows ONLY torch dtype/device).

  fidelity  - VALUE / SWAP / END / EXCEPTION / cross_invoker-push AND a tracer.cache()-style
              spec carrying ``torch.dtype`` + ``torch.device`` all round-trip exactly.
  security  - a frame whose data is a ``__reduce__`` gadget is REFUSED at decode
              (``find_class`` rejects ``os.system`` before the REDUCE could call it) — the
              gadget never executes on the host. A non-allowlisted class is also refused.

Needs torch; **CPU is enough — no CUDA required** (the bounce buffer is just bytes here).
Run:
  PYTHONPATH=src python \
    prototypes/mediator-sandbox/gpu_sandbox/test_isolated_codec_security.py
"""
import os
import pickle
import sys

import torch

from nnsight.intervention import transport as T
from nnsight.intervention.interleaver import Events


# Module-level (a function-local class can't be pickled even to ENCODE the frame, which
# would mask the decode-time refusal we mean to test). Harmless, but not on the allowlist.
class _NonAllowlisted:
    pass


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

    # SWAP: (requester, value) carrying a tensor.
    t = torch.randn(2, 3)
    ev, out, _ = _roundtrip(Events.SWAP, ("h.6.output", t))
    ok &= ev is Events.SWAP and out[0] == "h.6.output" and torch.equal(out[1], t)

    # END: saved dict — tensor + scalars + nested list/tuple + tuple-with-None.
    saved = {
        "x": torch.arange(6).float(),
        "n": 3,
        "f": 1.5,
        "lst": [1, (2, 3)],
        "tup": (torch.ones(2), None),
    }
    ev, out, _ = _roundtrip(Events.END, saved)
    ok &= ev is Events.END
    ok &= torch.equal(out["x"], saved["x"]) and out["n"] == 3 and out["f"] == 1.5
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

    # EXCEPTION: rebuilt host-side from (type-name, message); no object crosses.
    ev, out, _ = _roundtrip(Events.EXCEPTION, ValueError("boom"))
    ok &= ev is Events.EXCEPTION and isinstance(out, ValueError) and "boom" in str(out)

    # cross_invoker push: CPU tensors + scalars ride safetensors, not pickle.
    push = {"shared": torch.randn(4), "k": 7}
    ev, _out, pout = _roundtrip(Events.END, {"y": torch.zeros(1)}, push=push)
    ok &= pout is not None and torch.equal(pout["shared"], push["shared"]) and pout["k"] == 7

    print(f"[fidelity] VALUE/SWAP/END/CACHE(dtype,device)/EXCEPTION/push exact: {ok}")
    return ok


def test_security():
    probe = "/tmp/nnsight_codec_pwn_probe"
    if os.path.exists(probe):
        os.remove(probe)

    class Bomb:
        def __reduce__(self):
            return (os.system, (f"echo pwned > {probe}",))

    buf = torch.empty(1 << 16, dtype=torch.uint8)

    # The worker CAN pickle a gadget (encoding is safe — nothing runs). The host must
    # REFUSE it at decode (find_class rejects os.system) before the gadget executes.
    frame, _had = T._encode_worker_frame(Events.END, {"evil": Bomb()}, None, buf)
    refused = False
    try:
        T._decode_worker_frame(frame, buf)
    except pickle.UnpicklingError:
        refused = True
    no_pwn = not os.path.exists(probe)
    if os.path.exists(probe):
        os.remove(probe)

    # A non-allowlisted (but harmless) class is also refused — the allowlist is tight.
    frame2, _ = T._encode_worker_frame(Events.END, {"obj": _NonAllowlisted()}, None, buf)
    refused2 = False
    try:
        T._decode_worker_frame(frame2, buf)
    except pickle.UnpicklingError:
        refused2 = True

    ok = refused and no_pwn and refused2
    print(f"[security] gadget refused at decode + never executed: {refused and no_pwn} | "
          f"non-allowlisted class refused: {refused2}")
    return ok


def main():
    results = {"fidelity": test_fidelity(), "security": test_security()}
    ok = all(results.values())
    print("=" * 72)
    print(f"ISOLATED CODEC SECURITY: {'PASS' if ok else 'FAIL'} — {results}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

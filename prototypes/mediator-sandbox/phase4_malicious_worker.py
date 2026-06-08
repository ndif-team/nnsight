#!/usr/bin/env python3
"""Phase 4 — a MALICIOUS jailed tenant. Runs INSIDE the bwrap jail.

Attempts the mediator-capability leaks (test_mediator_capability.py #1-6,#8) that
SUCCEED in-process today: mutate its own batch_group to widen its slice, and walk
to the shared Interleaver/Batcher to read or poison sibling rows. In the isolated
design all of these are structurally inert — the jail's Mediator has no
interleaver/batcher reference, and the HOST owns the narrow bounds, so the worker
can only ever touch its own admitted row regardless of what it claims.

It reports what it tried + exactly what value it received (shape/sum), so the host
can prove the worker never saw the victim row, then poisons its own provider.
"""
import os
import socket


def main():
    fd = int(os.environ["WORKER_FD"])
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=fd)
    provider = os.environ["PROVIDER"]

    from types import SimpleNamespace

    from nnsight.intervention.interleaver import Mediator
    from nnsight.intervention.transport import SocketWorkerChannel, send_frame

    med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=None)
    med.channel = SocketWorkerChannel(sock)
    med.cross_invoker = False

    report = {}

    # leak #1/#2/#3/#8: try to widen this mediator's slice to grab the whole batch.
    for label, bg in [("None", None), ("sentinel[-1,0]", [-1, 0]), ("widen[0,2]", [0, 2])]:
        try:
            med.batch_group = bg
            report[f"set_batch_group_{label}"] = "set-on-local-mediator-only"
        except Exception as e:  # noqa: BLE001
            report[f"set_batch_group_{label}"] = "blocked:" + type(e).__name__
    med.batch_group = None  # leave it at the most-permissive claim before requesting

    # leak #4/#5/#6/#8: walk to the shared batcher / siblings / narrow (host objects).
    report["has_interleaver"] = med.interleaver is not None   # __init__ always sets it (to None)
    try:
        _ = med.interleaver.batcher.current_value
        report["batcher_walk"] = "REACHED-HOST-BATCHER"
    except Exception as e:  # noqa: BLE001
        report["batcher_walk"] = "CONTAINED:" + type(e).__name__
    try:
        _ = med.interleaver.mediators
        report["sibling_walk"] = "REACHED-SIBLINGS"
    except Exception as e:  # noqa: BLE001
        report["sibling_walk"] = "CONTAINED:" + type(e).__name__
    try:
        med.interleaver.batcher.narrow(None)   # leak #8: call narrow(None) directly
        report["direct_narrow"] = "REACHED-NARROW"
    except Exception as e:  # noqa: BLE001
        report["direct_narrow"] = "CONTAINED:" + type(e).__name__

    # Now actually request — the HOST narrows to THIS tenant's admitted row,
    # ignoring the None claim above.
    value = med.request(provider)
    hs = value[0] if isinstance(value, tuple) else value
    report["received_shape"] = list(hs.shape)
    report["received_sum"] = float(hs.sum())

    # leak #6: poison — but swap only lands on the host-recorded bounds (our row).
    poison = hs * 0.0 + 999.0
    med.swap(provider, (poison,) if isinstance(value, tuple) else poison)
    med.end()

    send_frame(sock, report)  # raw report frame, after the protocol completes


if __name__ == "__main__":
    main()

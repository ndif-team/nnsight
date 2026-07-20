#!/usr/bin/env python3
"""Phase 3 — the intervention worker, as exec'd INSIDE a bwrap jail.

Reconstructs the SocketWorkerChannel from an inherited fd and runs the real
Mediator client protocol (request -> double -> swap -> end) against the host.
In MODE=escape it ALSO runs the escape-suite gadgets first and ships a report
back (a raw frame, before the protocol) so the host can confirm they were
attempted-but-inert — while the legitimate protocol still completes through the
jail boundary.

Launched by phase3_jail_transport.py via:
  bwrap --unshare-all ... <python> phase3_jailed_worker.py
with env WORKER_FD / MODE / PROVIDER / SECRET / PWNED.
"""
import os
import socket


def run_escapes(secret_path, pwned_path):
    """Attempt host-affecting escapes; every one must be inert in the jail."""
    report = {}
    try:
        data = open(secret_path).read()
        report["fs_read"] = "LEAKED:" + data[:24]
    except Exception as e:
        report["fs_read"] = "CONTAINED:" + type(e).__name__

    # canonical __subclasses__ walk -> os -> os.system(touch host file)
    try:
        import subprocess  # noqa: F401  (populate the subclass graph)
        popen = None
        for c in ().__class__.__mro__[1].__subclasses__():
            if c.__name__ == "Popen" and c.__module__ == "subprocess":
                popen = c
                break
        osmod = popen.__init__.__globals__["os"]
        osmod.system("touch %s 2>/dev/null" % pwned_path)
        report["subclasses_os_write"] = "ran (host file checked by host)"
    except Exception as e:
        report["subclasses_os_write"] = "EXC:" + type(e).__name__

    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2).close()
        report["net_egress"] = "LEAKED"
    except Exception as e:
        report["net_egress"] = "CONTAINED:" + type(e).__name__

    return report


def main():
    fd = int(os.environ["WORKER_FD"])
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=fd)
    mode = os.environ.get("MODE", "double")
    provider = os.environ["PROVIDER"]

    if mode == "escape":
        report = run_escapes(os.environ["SECRET"], os.environ["PWNED"])
        from nnsight.intervention.transport import send_frame
        send_frame(sock, report)  # raw report frame, BEFORE the Mediator protocol

    # The legitimate protocol — identical to Phase 2, now from inside the jail.
    from types import SimpleNamespace

    from nnsight.intervention.interleaver import Mediator
    from nnsight.intervention.transport import SocketWorkerChannel

    med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=None)
    med.channel = SocketWorkerChannel(sock)
    med.cross_invoker = False

    value = med.request(provider)
    if isinstance(value, tuple):
        new_value = (value[0] * 2.0,) + tuple(value[1:])
    else:
        new_value = value * 2.0
    med.swap(provider, new_value)
    med.end()


if __name__ == "__main__":
    main()

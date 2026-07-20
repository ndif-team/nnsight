#!/usr/bin/env python3
"""Phase 6 — jailed worker using the shared-memory + safetensors channel.

Same as phase3's worker, but its tensor payloads ride a shared memfd (passed in
via SHM_FD) instead of being pickled over the socket. Runs INSIDE a bwrap jail.
"""
import os
import socket


def main():
    fd = int(os.environ["WORKER_FD"])
    shm_fd = int(os.environ["SHM_FD"])
    shm_size = int(os.environ["SHM_SIZE"])
    provider = os.environ["PROVIDER"]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=fd)

    from types import SimpleNamespace

    from nnsight.intervention.interleaver import Mediator
    from nnsight.intervention.transport import ShmArena, ShmSocketWorkerChannel

    arena = ShmArena.attach(shm_fd, shm_size)
    med = Mediator(intervention=None, info=SimpleNamespace(frame=None), batch_group=None)
    med.channel = ShmSocketWorkerChannel(sock, arena)
    med.cross_invoker = False

    value = med.request(provider)
    new = (value[0] * 2.0,) + tuple(value[1:]) if isinstance(value, tuple) else value * 2.0
    med.swap(provider, new)
    med.end()


if __name__ == "__main__":
    main()

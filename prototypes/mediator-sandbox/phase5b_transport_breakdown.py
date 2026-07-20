#!/usr/bin/env python3
"""Phase 5b — WHERE does the per-hook cost come from, and what fixes it.

Phase 5 showed ~111 ms/hook for a 16.8 MB activation, "dominated by sockRTT". But
a Unix socket moves GB/s, so 16.8 MB should cost a few ms — the 100 ms must be
SERIALIZATION, not transfer. This isolates it: it (1) measures the raw socket
bandwidth, (2) breaks the pickle path into dumps/loads, then compares three
round-trip transports on the SAME CPU tensor:

  A. pickle      — the naive path (pickle.dumps/​loads both ways)  [what Phase 5 used]
  B. raw message — a small struct header + the tensor's raw bytes over the socket,
                   rebuilt with torch.frombuffer (no pickle)
  C. shared mem  — host & worker share an mmap; only a 1-byte ready signal crosses
                   the socket; the bulk never travels (near zero-copy)

CPU<->CPU only (D2H/H2D is separate and was ~3-4 ms). Worker is a forked process.
Run:  PYTHONPATH=src .../hf-serve/bin/python prototypes/mediator-sandbox/phase5b_transport_breakdown.py
"""
import faulthandler
import mmap
import os
import pickle
import socket
import struct
import sys
import time

import torch

faulthandler.dump_traceback_later(45, exit=True)

DT = torch.bfloat16
SIZES = [(1, 512, 4096), (1, 2048, 4096), (1, 2048, 8192)]   # 4.2, 16.8, 33.6 MB
ITERS, WARM = 30, 8
_LEN = struct.Struct("!I")
_HDR = struct.Struct("!iiqqq")   # ndim, dtype_code(0=bf16), d0, d1, d2  (fixed 3-dim here)


def _recvn(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        c = sock.recv_into(view[got:], n - got)
        if c == 0:
            raise EOFError
        got += c
    return buf


# ---------- A: pickle ----------
def send_pickle(sock, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_LEN.pack(len(b)) + b)


def recv_pickle(sock):
    (n,) = _LEN.unpack(_recvn(sock, 4))
    return pickle.loads(_recvn(sock, n))


# ---------- B: raw message ----------
def send_raw(sock, t):
    t = t.contiguous()
    raw = t.flatten().view(torch.uint8).numpy()           # zero-copy view of the bytes
    hdr = _HDR.pack(t.dim(), 0, *(list(t.shape) + [0, 0, 0])[:3])
    sock.sendall(_LEN.pack(raw.nbytes) + hdr)
    sock.sendall(memoryview(raw))                          # bulk: one zero-copy send


def recv_raw(sock):
    (n,) = _LEN.unpack(_recvn(sock, 4))
    ndim, _code, d0, d1, d2 = _HDR.unpack(_recvn(sock, _HDR.size))
    buf = _recvn(sock, n)
    shape = [d0, d1, d2][:ndim]
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(DT).view(*shape)


# ---------- C: shared memory ----------
def make_shm(nbytes):
    return mmap.mmap(-1, nbytes)        # anonymous MAP_SHARED, inherited across fork


def shm_write(buf, t):
    t = t.contiguous()
    raw = t.flatten().view(torch.uint8).numpy()
    buf[: raw.nbytes] = memoryview(raw)


def shm_read(buf, nbytes, shape):
    return torch.frombuffer(memoryview(buf)[:nbytes], dtype=torch.uint8).view(DT).view(*shape).clone()


# --------------------------------------------------------------------------- #
def fork_worker(fn, *a):
    parent, child = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        parent.close()
        fn(child, *a)
        os._exit(0)
    child.close()
    return pid, parent


def bench(label, host_sock, do_round, stop):
    for _ in range(WARM):
        do_round()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        do_round()
    dt = (time.perf_counter() - t0) / ITERS * 1e3
    stop()
    return dt


def main():
    print(f"{'transport':<14}{'4.2 MB':>12}{'16.8 MB':>12}{'33.6 MB':>12}   (ms / round-trip)")

    # raw socket bandwidth + pickle sub-breakdown, for the 16.8 MB case
    mid = torch.randn(*SIZES[1], dtype=DT)
    mid_bytes = mid.flatten().view(torch.uint8).numpy().nbytes
    # one-way raw socket throughput
    pid, hs = fork_worker(_drain, mid_bytes)
    raw = mid.flatten().view(torch.uint8).numpy()
    for _ in range(WARM):
        hs.sendall(memoryview(raw)); hs.recv(1)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        hs.sendall(memoryview(raw)); hs.recv(1)
    one_way = (time.perf_counter() - t0) / ITERS * 1e3
    hs.sendall(b""); hs.close(); os.waitpid(pid, 0)
    gbps = mid_bytes / (one_way / 1e3) / 1e9
    blob = pickle.dumps(mid, protocol=pickle.HIGHEST_PROTOCOL)
    dumps_ms = _t(lambda: pickle.dumps(mid, protocol=pickle.HIGHEST_PROTOCOL))
    loads_ms = _t(lambda: pickle.loads(blob))
    op_bf16 = _t(lambda: mid * 2.0)
    mid_f32 = mid.float()
    op_f32 = _t(lambda: mid_f32 * 2.0)
    print(f"\n[probe] 16.8 MB: raw socket one-way+ack {one_way:.2f} ms ({gbps:.1f} GB/s)")
    print(f"[probe] 16.8 MB: pickle.dumps {dumps_ms:.2f} ms | pickle.loads {loads_ms:.2f} ms")
    print(f"[probe] 16.8 MB: ×2 CPU op — bf16 {op_bf16:.2f} ms | fp32 {op_f32:.2f} ms "
          f"(this is the USER op, not transport)\n")

    # three transports across the three sizes — ECHO (no op) = PURE transport cost
    for name, runner in [("A pickle", run_pickle), ("B raw-msg", run_raw), ("C shared-mem", run_shm)]:
        cells = []
        for sz in SIZES:
            cells.append(f"{runner(sz):>12.2f}")
        print(f"{name:<14}{''.join(cells)}")

    print("\n(worker does the same ×2 CPU op in every method; only the transport differs.)")


def _drain(sock, nbytes):
    while True:
        try:
            b = _recvn(sock, nbytes)
        except EOFError:
            return
        sock.sendall(b"k")


def _t(fn, n=20):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


def run_pickle(sz):
    def worker(sock):
        while True:
            try:
                t = recv_pickle(sock)
            except EOFError:
                return
            send_pickle(sock, t)  # echo = pure transport
    pid, hs = fork_worker(worker)
    act = torch.randn(*sz, dtype=DT)

    def rnd():
        send_pickle(hs, act); recv_pickle(hs)
    dt = bench("A", hs, rnd, lambda: (hs.close(), os.waitpid(pid, 0)))
    return dt


def run_raw(sz):
    def worker(sock):
        while True:
            try:
                t = recv_raw(sock)
            except EOFError:
                return
            send_raw(sock, t)  # echo = pure transport
    pid, hs = fork_worker(worker)
    act = torch.randn(*sz, dtype=DT)

    def rnd():
        send_raw(hs, act); recv_raw(hs)
    dt = bench("B", hs, rnd, lambda: (hs.close(), os.waitpid(pid, 0)))
    return dt


def run_shm(sz):
    nbytes = torch.empty(*sz, dtype=DT).flatten().view(torch.uint8).numpy().nbytes
    in_buf = make_shm(nbytes)
    out_buf = make_shm(nbytes)

    def worker(sock):
        while True:
            sig = sock.recv(1)
            if not sig:
                return
            t = torch.frombuffer(memoryview(in_buf)[:nbytes], dtype=torch.uint8).view(DT).view(*sz)
            res = t.contiguous()  # echo = pure transport
            out_buf[:nbytes] = memoryview(res.flatten().view(torch.uint8).numpy())
            sock.sendall(b"k")

    pid, hs = fork_worker(worker)
    act = torch.randn(*sz, dtype=DT)

    def rnd():
        shm_write(in_buf, act)
        hs.sendall(b"g")
        hs.recv(1)
        shm_read(out_buf, nbytes, sz)
    dt = bench("C", hs, rnd, lambda: (hs.close(), os.waitpid(pid, 0)))
    in_buf.close(); out_buf.close()
    return dt


if __name__ == "__main__":
    main()

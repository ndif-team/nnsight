"""Socket-backed :class:`MediatorChannel` implementations for running an
intervention worker in a *separate process* from the model forward pass.

This is the socket-transport stage of the mediator-isolation work
(docs/developing/mediator-isolation-harness-plan.md): the same six-event protocol
(VALUE/SWAP/SKIP/BARRIER/END/EXCEPTION), but the worker<->main handoff rides on an
``AF_UNIX`` socket instead of the in-process one-slot queues.

Two role-specific ends, because each side of the protocol only ever calls its own
half of :class:`~nnsight.intervention.interleaver.MediatorChannel`:

- **Host** (main thread, model forward): calls ``wait_event`` / ``has_event`` /
  ``get_event`` / ``restore_event`` / ``put_response``.  Worker events arrive over
  the socket into a *local one-slot buffer*; ``has_event`` / ``get_event`` /
  ``restore_event`` operate on that buffer with **no wire traffic** (they mirror the
  in-process flag/restore semantics), and only ``wait_event`` reads the socket.
  This is required: ``restore_event`` re-stages an event the main thread will
  consume on a later ``handle()`` pass, which is host-local, not a round-trip.
- **Worker** (intervention fn): calls ``put_event`` / ``wait_response`` /
  ``get_response``.  ``put_event`` sends a frame; ``wait_response`` blocks reading
  the reply.

Frame codec: length-prefixed ``pickle``.  This socket path forks the worker, so both ends
are mutually trusted and pickle-both-ways is fine.  Two follow-ups (see the plan):

1. **Security:** once the worker is untrusted, the *jail->host* direction MUST NOT
   ``pickle.loads`` arbitrary objects — restrict the host-side decoder to
   tensors/known frames.  (jail<-host is host-authored, so the jail trusting it is ok.)

2. **Performance (measured):** ``pickle`` of a torch tensor is the per-hook
   bottleneck — ``dumps``+``loads`` ~22 ms per direction at 16.8 MB and **superlinear**;
   a 16.8 MB round-trip is ~96 ms vs ~10 ms of actual socket transfer.  Swapping the
   codec to a **raw header + bytes** form (``memoryview``/``torch.frombuffer``, i.e.
   ``safetensors``) is ~4×; a **shared-memory ring** (bulk never crosses the socket) is
   ~8× and linear.  This codec is the thing to replace, not the boundary.
"""

from __future__ import annotations

import mmap
import os
import pickle
import socket
import struct
from typing import Any, Optional

import torch

from .interleaver import MediatorChannel

try:
    from safetensors.torch import load as _st_load
    from safetensors.torch import save as _st_save

    _HAS_SAFETENSORS = True
except Exception:  # pragma: no cover
    _HAS_SAFETENSORS = False

_HEADER = struct.Struct("!I")  # 4-byte big-endian length prefix


def _recvn(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise ``EOFError`` if the peer closed."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("mediator channel closed by peer")
        buf += chunk
    return bytes(buf)


def send_frame(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv_frame(sock: socket.socket) -> Any:
    (n,) = _HEADER.unpack(_recvn(sock, _HEADER.size))
    return pickle.loads(_recvn(sock, n))


class SocketHostChannel(MediatorChannel):
    """Host (main-thread) end of the channel — lives in the model process.

    Worker events are read off the socket into a single-slot buffer; the
    main-thread protocol then drains that buffer exactly as it drained the
    in-process event queue.  Only :meth:`wait_event` touches the socket.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._pending: Any = None
        self._has = False

    # --- worker -> main (read from socket into the local buffer) ---
    def wait_event(self) -> None:
        # Block for the worker's next event, unless one is already buffered
        # (e.g. just restored by a provider-mismatch on this handle pass).
        if not self._has:
            self._pending = recv_frame(self._sock)
            self._has = True

    @property
    def has_event(self) -> bool:
        return self._has

    def get_event(self) -> Any:
        item = self._pending
        self._pending = None
        self._has = False
        return item

    def restore_event(self, item: Any) -> None:
        # Host-local re-stage (no wire traffic) — the main thread consumes this
        # on a later handle() pass when the matching provider fires.
        self._pending = item
        self._has = True

    # --- main -> worker (send the reply for a pending event) ---
    def put_response(self, value: Any) -> None:
        send_frame(self._sock, value)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class SocketWorkerChannel(MediatorChannel):
    """Worker (intervention-fn) end — lives in the separate worker process.

    The worker only ever does ``send -> wait -> get``: push an event, block for
    the reply, take it.  No host-side staging here.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._response: Any = None
        self._has_response = False

    # --- worker -> main (send the event) ---
    def put_event(self, item: Any) -> None:
        send_frame(self._sock, item)

    # --- main -> worker (block for the reply) ---
    def wait_response(self) -> None:
        self._response = recv_frame(self._sock)
        self._has_response = True

    def get_response(self) -> Any:
        item = self._response
        self._response = None
        self._has_response = False
        return item

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


# =========================================================================== #
# Fast path: shared memory + safetensors                                      #
# =========================================================================== #
# Measurement showed the per-hook cost is pickle, not the boundary: pickling a torch
# tensor is ~22 ms per direction at 16.8 MB (superlinear), while the socket moves
# the bytes in ~10 ms. This path fixes both:
#   - tensor BULK travels through a shared-memory region (memfd), so it never
#     crosses the socket — only a tiny control frame does;
#   - tensors are encoded with **safetensors** (a safe, no-code-execution format),
#     which also closes the jail->host untrusted-deserialize hole for the bulk.
#
# Still pickle: the small CONTROL frame (the non-tensor structure + the requester
# string + the byte length). Hardening that to a restricted decoder is the
# remaining jail->host security item (see the module header / the plan).

_TENSOR_TAG = "__nnsight_shm_t__"


def _split_tensors(obj: Any, store: dict) -> Any:
    """Walk ``obj``; pull every tensor into ``store`` (keyed by index), leaving a
    ``{_TENSOR_TAG: i}`` placeholder. Non-tensor structure is returned as-is."""
    if torch.is_tensor(obj):
        i = len(store)
        store[str(i)] = obj.detach().contiguous().cpu()
        return {_TENSOR_TAG: i}
    if type(obj) is tuple:
        return tuple(_split_tensors(x, store) for x in obj)
    if type(obj) is list:
        return [_split_tensors(x, store) for x in obj]
    if type(obj) is dict:
        return {k: _split_tensors(v, store) for k, v in obj.items()}
    return obj


def _merge_tensors(skel: Any, tensors: dict) -> Any:
    """Reverse of :func:`_split_tensors` — re-inject tensors into the skeleton."""
    if type(skel) is dict:
        if len(skel) == 1 and _TENSOR_TAG in skel:
            return tensors[str(skel[_TENSOR_TAG])]
        return {k: _merge_tensors(v, tensors) for k, v in skel.items()}
    if type(skel) is tuple:
        return tuple(_merge_tensors(x, tensors) for x in skel)
    if type(skel) is list:
        return [_merge_tensors(x, tensors) for x in skel]
    return skel


class ShmArena:
    """A shared-memory region (anonymous ``memfd``) for moving tensor payloads
    out-of-band of the control socket.

    The host creates one; the worker attaches to the SAME memfd via an inherited
    fd (``os.fork``) or one passed into the jail (``pass_fds`` + ``SHM_FD`` env),
    and both ``mmap`` it. Under the one-event-in-flight protocol a single region
    is reused in strict alternation; the receiver always ``safetensors.load``s
    (which copies into fresh tensors) before the next write, so there is no
    aliasing between the live tensors and the buffer.
    """

    def __init__(self, size: int, fd: int = None, owns: bool = True):
        if fd is None:
            fd = os.memfd_create("nnsight-shm", 0)
            os.ftruncate(fd, size)
        self.fd = fd
        self.size = size
        self._owns = owns
        self.buf = mmap.mmap(fd, size)

    @classmethod
    def attach(cls, fd: int, size: int) -> "ShmArena":
        """Attach to an existing memfd (inherited / passed-in) without owning it."""
        return cls(size, fd=fd, owns=False)

    def write(self, blob: bytes) -> int:
        n = len(blob)
        if n > self.size:
            raise ValueError(f"payload {n} B exceeds shm arena {self.size} B")
        self.buf[:n] = blob
        return n

    def read(self, n: int) -> bytes:
        return bytes(self.buf[:n])

    def close(self) -> None:
        try:
            self.buf.close()
            if self._owns:
                os.close(self.fd)
        except (OSError, ValueError):
            pass


def send_shm(sock: socket.socket, arena: ShmArena, obj: Any) -> None:
    """Write ``obj``'s tensors (safetensors) into ``arena`` and send only the
    small control skeleton + byte length over the socket."""
    store: dict = {}
    skel = _split_tensors(obj, store)
    n = arena.write(_st_save(store)) if store else 0
    ctrl = pickle.dumps((skel, n), protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(ctrl)) + ctrl)


def recv_shm(sock: socket.socket, arena: ShmArena) -> Any:
    (clen,) = _HEADER.unpack(_recvn(sock, _HEADER.size))
    skel, n = pickle.loads(_recvn(sock, clen))
    tensors = _st_load(arena.read(n)) if n else {}
    return _merge_tensors(skel, tensors)


# =========================================================================== #
# Fast path: GPU bounce buffer (CUDA IPC)                                      #
# =========================================================================== #
# The chosen design (docs/developing/gpu-sandbox.md + mediator-gpu-trace-
# integration.md): the worker is GPU-enabled and shares ONE GPU "bounce buffer"
# with the host via CUDA IPC (mapped once before lockdown). Tensor bulk never
# leaves the GPU and never hits pickle — only a small control frame (the
# non-tensor skeleton + an offset table) crosses the pipe.
#
# Under the one-event-in-flight protocol the single buffer is reused in strict
# alternation, so the receiver MUST clone tensors out of the buffer before the
# next write can overwrite them (the clone-on-receive rule). A D2D clone is
# HBM-speed; "zero-copy" means no-PCIe-no-pickle, not no-copy.


def _dtype_from_str(s: str) -> torch.dtype:
    """``"torch.bfloat16"`` -> ``torch.bfloat16`` (no eval, no getattr-default)."""
    return getattr(torch, s.split(".")[-1])


def pack_cuda(value: Any, buf: torch.Tensor) -> tuple:
    """Write every tensor in ``value`` into the shared GPU ``buf`` (D2D), leaving a
    non-tensor skeleton + an offset table describing where each tensor landed.

    Returns ``(skeleton, table)`` where ``table[str(i)] = (offset, nbytes, shape,
    dtype_str)``. Offsets are 16-byte aligned so a ``uint8`` slice can be viewed as
    any tensor dtype. Raises if the payload exceeds the arena.
    """
    table: dict = {}
    state = {"offset": 0}

    def walk(obj: Any) -> Any:
        if torch.is_tensor(obj):
            i = len(table)
            t = obj.detach().contiguous()
            flat = t.reshape(-1).view(torch.uint8)
            n = int(flat.numel())
            offset = (state["offset"] + 15) & ~15  # 16-byte align
            if offset + n > buf.numel():
                raise ValueError(
                    f"intervention value {offset + n} B exceeds GPU bounce buffer "
                    f"{buf.numel()} B"
                )
            if n:
                buf[offset : offset + n].copy_(flat)
            table[str(i)] = (offset, n, tuple(t.shape), str(t.dtype))
            state["offset"] = offset + n
            return {_TENSOR_TAG: i}
        if type(obj) is tuple:
            return tuple(walk(x) for x in obj)
        if type(obj) is list:
            return [walk(x) for x in obj]
        if type(obj) is dict:
            return {k: walk(v) for k, v in obj.items()}
        return obj

    skel = walk(value)
    return skel, table


def unpack_cuda(skel: Any, table: dict, buf: torch.Tensor) -> Any:
    """Reverse of :func:`pack_cuda`. Each tensor is CLONED out of ``buf`` (so a
    later reuse of the single buffer can't corrupt it — the clone-on-receive
    rule) and re-injected into the skeleton."""
    tensors: dict = {}
    for k, (offset, n, shape, dtype_str) in table.items():
        dtype = _dtype_from_str(dtype_str)
        view = buf[offset : offset + n].view(dtype).reshape(shape)
        tensors[k] = view.clone()
    return _merge_tensors(skel, tensors)


class CudaIpcHostChannel(MediatorChannel):
    """Host (main-thread) end of the GPU-bounce-buffer channel.

    Control frames ride an ``mp.Connection``; tensor bulk rides the shared CUDA
    ``buf`` (mapped into the worker via CUDA IPC). Same one-event-in-flight buffer
    semantics as :class:`SocketHostChannel`: worker events arrive on ``wait_event``
    and are unpacked (cloned) into a single-slot local buffer; ``has_event`` /
    ``get_event`` / ``restore_event`` are host-local with no IPC.
    """

    def __init__(
        self,
        conn: Any,
        buf: torch.Tensor,
        timeout: Optional[float] = None,
        startup_timeout: float = 180.0,
    ):
        self._conn = conn
        self._buf = buf
        self._pending: Any = None
        self._has = False
        # Per-wait timeout for a hung worker (infinite loop in user code). The
        # FIRST event covers spawn + import + deserialize + run-to-first-request,
        # which is slow, so it gets a generous startup_timeout; subsequent waits
        # use the user timeout. On timeout the trace's ``finally: cancel()`` kills
        # the worker (interleave -> cancel -> _iso.close).
        self._timeout = timeout
        self._startup_timeout = startup_timeout
        self._started = False
        # Optional callable -> dict of live host interleaver state to piggyback on
        # each response (e.g. ``default_all``, set only after the worker spawns).
        self.meta_provider = None
        # Optional callable(dict) to merge a worker's pushed cross_invoker locals
        # into the host-side shared variable store. None on the in-process path.
        self.on_push = None

    # --- worker -> main ---
    def wait_event(self) -> None:
        if not self._has:
            limit = self._startup_timeout if not self._started else self._timeout
            if limit is not None and not self._conn.poll(limit):
                raise TimeoutError(
                    f"sandboxed intervention exceeded {limit}s with no event "
                    f"— worker presumed hung (e.g. an infinite loop in user code)"
                )
            try:
                event, skel, table, push = self._conn.recv()
            except (EOFError, OSError) as e:
                # The pipe broke mid-protocol => the worker died (e.g. a segfault
                # in user C-code, or the GPU process was OOM-killed). Surface a
                # clean error so the trace's ``finally: cancel()`` tears down,
                # instead of leaking a raw EOFError out of model.trace().
                raise RuntimeError(
                    "sandboxed intervention worker died during execution"
                ) from e
            if push is not None and self.on_push is not None:
                self.on_push(push)  # cross_invoker: merge into the host var store
            self._started = True
            self._pending = (event, unpack_cuda(skel, table, self._buf))
            self._has = True

    @property
    def has_event(self) -> bool:
        return self._has

    def get_event(self) -> Any:
        item = self._pending
        self._pending = None
        self._has = False
        return item

    def restore_event(self, item: Any) -> None:
        self._pending = item
        self._has = True

    def reset(self) -> None:
        """Clear single-slot + per-job state so a recycled worker's channel is fresh.

        Re-arms the generous startup timeout (the next job's first event covers a
        fresh deserialize + run-to-first-request) and drops the per-mediator
        ``meta_provider`` / ``on_push`` bindings (re-set on the next acquire)."""
        self._pending = None
        self._has = False
        self._started = False
        self.meta_provider = None
        self.on_push = None

    # --- main -> worker ---
    def put_response(self, value: Any) -> None:
        skel, table = pack_cuda(value, self._buf)
        if table:
            # pack_cuda's D2D copies are async on this context's stream; the Pipe
            # only orders the CPU side. Without this sync the WORKER's unpack_cuda
            # clone (a SEPARATE CUDA context) can read the buffer before our copy
            # has run device-side -> silent corruption. (The proven prototype
            # gpu_sandbox.py:43 / gpu_worker.py:57 synchronized here; the port
            # dropped it — tests passed only because gpt2 tensors are tiny.)
            torch.cuda.synchronize()
        meta = self.meta_provider() if self.meta_provider is not None else None
        self._conn.send((skel, table, meta))

    def close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass


class CudaIpcWorkerChannel(MediatorChannel):
    """Worker (intervention-fn) end of the GPU-bounce-buffer channel.

    Only ever does ``put_event -> wait_response -> get_response``. Tensors in the
    outgoing event payload are written into the shared buffer; the response's
    tensors are read (cloned) back out of it.
    """

    def __init__(self, conn: Any, buf: torch.Tensor):
        self._conn = conn
        self._buf = buf
        self._response: Any = None
        self._has_response = False
        # Optional callable(dict) applied to piggybacked host state on each response.
        self.on_meta = None
        # Optional callable() -> dict of cross_invoker locals to push to the host
        # store on each event (None / returns None on non-cross-invoke traces).
        self.push_provider = None

    # --- worker -> main ---
    def put_event(self, item: Any) -> None:
        event, data = item
        skel, table = pack_cuda(data, self._buf)
        if table:
            # Async D2D copies must finish before the HOST (separate CUDA context)
            # clones them out of the buffer. See CudaIpcHostChannel.put_response.
            torch.cuda.synchronize()
        push = self.push_provider() if self.push_provider is not None else None
        self._conn.send((event, skel, table, push))

    # --- main -> worker ---
    def wait_response(self) -> None:
        skel, table, meta = self._conn.recv()
        if meta is not None and self.on_meta is not None:
            self.on_meta(meta)
        self._response = unpack_cuda(skel, table, self._buf)
        self._has_response = True

    def get_response(self) -> Any:
        item = self._response
        self._response = None
        self._has_response = False
        return item

    def close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass


class ShmSocketHostChannel(SocketHostChannel):
    """Host channel whose tensor payloads ride a :class:`ShmArena` instead of the
    socket. Same one-event-in-flight buffer semantics as the parent."""

    def __init__(self, sock: socket.socket, arena: ShmArena):
        super().__init__(sock)
        self._arena = arena

    def wait_event(self) -> None:
        if not self._has:
            self._pending = recv_shm(self._sock, self._arena)
            self._has = True

    def put_response(self, value: Any) -> None:
        send_shm(self._sock, self._arena, value)


class ShmSocketWorkerChannel(SocketWorkerChannel):
    """Worker channel whose tensor payloads ride a :class:`ShmArena`."""

    def __init__(self, sock: socket.socket, arena: ShmArena):
        super().__init__(sock)
        self._arena = arena

    def put_event(self, item: Any) -> None:
        send_shm(self._sock, self._arena, item)

    def wait_response(self) -> None:
        self._response = recv_shm(self._sock, self._arena)
        self._has_response = True

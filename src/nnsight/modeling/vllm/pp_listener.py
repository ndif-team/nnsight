"""PP Listener — cross-rank tensor pull via a dedicated gloo process group.

Concurrency-safe: several mediator worker threads on one rank may each issue
their own pull at the same time. gloo routes point-to-point traffic strictly by
``(peer, tag)`` (never by content) and a blocking send/recv is an ordered
rendezvous, so the protocol is built around two rules:

  - **Requests** are ONE fixed-size, self-identifying message on the shared
    ``TAG_REQUEST``. A single atomic message can't interleave with another
    consumer's request (a two-message header+key request would, and a
    size-mismatched recv aborts the worker). It carries the requester rank, a
    per-pull response tag, the mode flag, and the lookup key.
  - **Responses** ride a **per-pull response tag** (``TAG_RESPONSE_BASE + n``)
    carried in the request, so concurrent consumers each receive only their own
    reply. The consumer's recv on that tag is posted before the producer sends
    (it's the next thing the pulling thread does), satisfying the rendezvous.

The producer's listener is a single thread serving requests in arrival order.

Buffers store narrowed (per-mediator) tensors on GPU; moved to CPU at pull time.
Dtype and shape are resolved locally from a shared metadata map built at model
load time — no metadata on the wire for modules with known shapes.  Modules
without metadata fall back to a legacy protocol that sends shape info (on the
same per-pull tag).
"""

from __future__ import annotations

import itertools
import struct
import threading
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .pp import resolve_meta as _resolve_meta

# Requests all arrive on this one well-known tag (the listener can only
# pre-post a recv on a tag it knows). Responses ride a per-pull tag carried
# IN the request (see ``_encode_request``), starting at ``TAG_RESPONSE_BASE``
# so they never alias ``TAG_REQUEST``. gloo routes strictly by ``(peer, tag)``
# and never by content, so a distinct response tag per in-flight pull is what
# lets concurrent consumers (multiple mediator threads on one rank) each get
# their own reply without cross-delivery — see ``pull_from_remote``.
TAG_REQUEST = 0
TAG_RESPONSE_BASE = 1024
# Response tags cycle through this range; with short-lived pulls a tag is only
# reused after a full cycle, never while one is concurrently in flight.
_TAG_RANGE = 1 << 20
_META_SLOTS = 32  # legacy metadata buffer size

# Separator between req_id and provider in the wire-encoded key.
# Format: ``"{req_id}|{provider}"`` (null-terminated).  req_id may be
# empty for requests that predate the composite-key discipline — the
# listener then falls back to the string-only provider key (legacy).
_KEY_SEP = "|"

# A pull request is ONE fixed-size, self-identifying message on TAG_REQUEST:
# a two-message (header-then-key) request would interleave under concurrent
# senders and the producer would pair one consumer's header with another's key
# (size mismatch → gloo aborts the worker, or a garbled lookup key wedges it).
# Layout: 4 little-endian int64 [requester_rank, response_tag, header_num_tokens,
# key_len] followed by the UTF-8 key bytes, zero-padded to a constant size so
# the listener's single recv always matches the send size.
_REQUEST_HEADER = struct.Struct("<4q")
REQUEST_MSG_BYTES = 256


def _encode_request(requester_rank, response_tag, header_num_tokens, wire_key):
    """Pack a pull request into one fixed-size ``REQUEST_MSG_BYTES`` uint8 buffer."""
    key_bytes = wire_key.encode("utf-8")
    capacity = REQUEST_MSG_BYTES - _REQUEST_HEADER.size
    if len(key_bytes) > capacity:
        raise ValueError(
            f"PP pull key {wire_key!r} is {len(key_bytes)}B; exceeds the "
            f"{capacity}B request-buffer capacity. Raise REQUEST_MSG_BYTES."
        )
    buf = bytearray(REQUEST_MSG_BYTES)
    _REQUEST_HEADER.pack_into(
        buf, 0, requester_rank, response_tag, header_num_tokens, len(key_bytes)
    )
    buf[_REQUEST_HEADER.size:_REQUEST_HEADER.size + len(key_bytes)] = key_bytes
    # bytearray is writable, so frombuffer doesn't warn; clone to own the memory.
    return torch.frombuffer(buf, dtype=torch.uint8).clone()


def _decode_request(buf):
    """Inverse of :func:`_encode_request`.

    Returns ``(requester_rank, response_tag, header_num_tokens, wire_key)``.
    """
    raw = bytes(buf.numpy())
    requester_rank, response_tag, header_num_tokens, key_len = (
        _REQUEST_HEADER.unpack_from(raw, 0)
    )
    wire_key = raw[
        _REQUEST_HEADER.size:_REQUEST_HEADER.size + key_len
    ].decode("utf-8")
    return requester_rank, response_tag, header_num_tokens, wire_key


class PPListener:
    """Cross-rank tensor pull service.

    Producer (background listener thread): recvs fixed-size requests on
    TAG_REQUEST, sends each reply on the per-pull tag carried in its request.

    Consumer (mediator thread): allocates a per-pull tag, sends one fixed-size
    request on TAG_REQUEST, recvs its reply on that tag. Many mediator threads
    may do this concurrently.

    The request packs ``[requester_rank, response_tag, header_num_tokens,
    key_len] + key`` (see ``_encode_request``). ``header_num_tokens > 0`` ⇒ the
    consumer pre-computed the recv buffer size from shared metadata, producer
    sends flat data only; ``== 0`` ⇒ legacy mode, producer sends shape metadata
    then flat data (both on the per-pull tag).

    Concurrent consumers never collide: each request is one atomic message and
    each reply rides a distinct tag.
    """

    def __init__(
        self,
        buffer: Dict[str, Any],
        condition: threading.Condition,
        pull_group: Optional[dist.ProcessGroup],
        local_rank: int,
        device: torch.device,
        meta_map: Optional[Dict[str, dict]] = None,
    ):
        self._buffer = buffer
        self._condition = condition
        self._pull_group = pull_group
        self._local_rank = local_rank
        self._device = device
        self._meta_map = meta_map or {}
        self._thread: Optional[threading.Thread] = None
        # Set by ``stop()`` to break the listen loop cleanly. Also
        # checked after every caught exception so a torn-down ``dist``
        # context (e.g. when the worker process is being shut down)
        # doesn't busy-loop the listener at 100% CPU.
        self._stop_event = threading.Event()
        # Allocates a distinct response tag per in-flight pull so concurrent
        # consumer threads each receive their own reply (gloo routes by
        # ``(peer, tag)``, never by content). ``itertools.count().__next__`` is
        # atomic under the GIL, so multiple mediator threads can allocate
        # without a lock. See ``_next_response_tag`` / ``pull_from_remote``.
        self._tag_counter = itertools.count()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def clear_buffer(self, req_ids=None):
        """Drop buffer entries and wake pending lookups.

        Args:
            req_ids: Optional iterable of request ids.  When supplied,
                only keys whose 2-tuple tail equals one of these ids
                are removed — in-flight requests' entries stay intact.
                When ``None``, falls back to the legacy blanket clear
                (all keys removed).

        Scoped clear is the Bug B companion fix: with composite keys
        ``(provider, req_id)``, a blanket clear at ANY request's
        completion would also wipe concurrent in-flight requests'
        slices and send their still-pending cross-rank pulls into a
        30 s timeout.

        We notify_all so waiters whose key was removed stop spinning
        immediately; waiters on untouched keys see their key still
        present after the wake and continue normally.
        """
        with self._condition:
            if req_ids is None:
                self._buffer.clear()
            else:
                id_set = set(req_ids)
                to_remove = [
                    k for k in self._buffer
                    if isinstance(k, tuple) and len(k) == 2 and k[1] in id_set
                ]
                for k in to_remove:
                    del self._buffer[k]
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Local buffer lookup (blocks until value available)
    # ------------------------------------------------------------------

    def local_lookup(
        self,
        key,
        timeout: Optional[float] = 60.0,
    ) -> torch.Tensor:
        """Block until ``key`` is in the buffer, then return its value.

        ``key`` is whatever the producer wrote under — for Bug B the
        producer writes ``(provider, req_id)`` composite tuples, so
        ``key`` here is a 2-tuple.  String keys are still supported for
        legacy callers and unit tests.
        """
        with self._condition:
            while key not in self._buffer:
                if not self._condition.wait(timeout=timeout):
                    raise TimeoutError(
                        f"PPListener: timed out waiting for {key!r}"
                    )
            return self._buffer[key]

    # ------------------------------------------------------------------
    # Producer: background listener thread
    # ------------------------------------------------------------------

    def start(self):
        if self._pull_group is None:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="pp-listener"
        )
        self._thread.start()

    def stop(self):
        """Signal the listener thread to exit on its next loop iteration."""
        self._stop_event.set()

    def _listen_loop(self):
        group = self._pull_group
        world_size = dist.get_world_size(group)
        other_ranks = [r for r in range(world_size) if r != self._local_rank]

        while not self._stop_event.is_set():
            try:
                # 1. Recv ONE fixed-size, self-identifying request on
                #    TAG_REQUEST. A single message (vs. the old header+key pair)
                #    is what makes concurrent senders safe: each request is
                #    atomic on the wire, so two mediators' requests can't
                #    interleave into a size-mismatch abort. ``group_src`` stays
                #    wildcard for PP>2 (any rank may request); the request
                #    itself carries ``requesting_rank`` for the reply.
                req_buf = torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8)
                src = other_ranks[0] if len(other_ranks) == 1 else None
                dist.recv(req_buf, group=group, group_src=src, tag=TAG_REQUEST)

                requesting_rank, response_tag, num_tokens, encoded = (
                    _decode_request(req_buf)
                )

                # Wire key is ``"{req_id}|{provider}"`` (Bug B composite key);
                # a missing separator (legacy callers / tests) is a plain
                # provider string with req_id=None.
                if _KEY_SEP in encoded:
                    req_id_str, provider_string = encoded.split(_KEY_SEP, 1)
                    lookup_key = (provider_string, req_id_str or None)
                else:
                    provider_string = encoded
                    lookup_key = encoded

                # 3. Look up value in buffer (blocks until available)
                value = self.local_lookup(lookup_key)

                # Normalize to list of tensors (handles both tensor and tuple)
                tensors = list(value) if isinstance(value, (tuple, list)) else [value]
                cpu_tensors = [t.detach().contiguous().cpu() for t in tensors]

                if num_tokens == 0:
                    # Legacy mode: send shape metadata then data.
                    shape_meta = torch.zeros(_META_SLOTS, dtype=torch.int64)
                    shape_meta[0] = len(cpu_tensors)
                    idx = 1
                    for t in cpu_tensors:
                        shape_meta[idx] = t.ndim
                        idx += 1
                        for s in t.shape:
                            shape_meta[idx] = s
                            idx += 1
                    dist.send(shape_meta, group=group, group_dst=requesting_rank, tag=response_tag)

                # 4. Send all tensor data concatenated as one flat buffer, on the
                #    per-pull response tag carried in the request — so concurrent
                #    consumers each receive only their own reply.
                flat = torch.cat([t.contiguous().view(-1) for t in cpu_tensors])
                dist.send(flat, group=group, group_dst=requesting_rank, tag=response_tag)

            except Exception:
                # If the dist context is gone (worker tearing down, peer
                # crashed) every ``dist.recv`` here would raise instantly
                # and the loop would burn CPU at 100% printing tracebacks
                # forever. Detect that case and exit cleanly. Genuine
                # transient errors still get logged and retried after a
                # short backoff so the listener stays responsive without
                # spinning.
                if not dist.is_initialized() or self._stop_event.is_set():
                    return
                import traceback
                traceback.print_exc()
                # Wait with timeout so ``stop()`` from the main thread
                # wakes us promptly; expire-then-retry on its own keeps
                # the listener alive through transient failures.
                if self._stop_event.wait(timeout=0.5):
                    return

    # ------------------------------------------------------------------
    # Consumer: pull tensor from remote rank
    # ------------------------------------------------------------------

    def pull_from_remote(
        self,
        source_rank: int,
        provider_string: str,
        num_tokens: int = 0,
        req_id: Optional[str] = None,
    ):
        """Pull tensor(s) from a remote rank.

        Args:
            source_rank: PP rank that owns the module.
            provider_string: Module-level provider (``"…output.iN"``).
            num_tokens: Number of tokens for this request (from scheduler,
                same on all PP ranks).  When > 0 and metadata is available,
                the recv buffer is pre-computed — no shape on the wire.
                When 0, falls back to legacy protocol with metadata.
            req_id: Bug B fix — vLLM request id.  When non-None, the wire
                key encodes ``"{req_id}|{provider}"`` and the producer
                looks up the composite ``(provider, req_id)`` tuple so
                concurrent mediators can't deliver each other's slices.
                ``None`` falls back to the legacy string-only key (used
                by unit tests and any non-vLLM caller).
        """
        if self._pull_group is None:
            raise RuntimeError("No pull_group configured for cross-rank pull")

        group = self._pull_group
        module_path = _provider_to_module_path(provider_string)
        meta = _resolve_meta(self._meta_map, module_path)

        # Decide protocol mode: optimized (pre-computed shapes) or legacy.
        use_precomputed = self._should_use_precomputed(meta, num_tokens)
        header_num_tokens = num_tokens if use_precomputed else 0

        # Wire-encode the key with req_id if available (Bug B composite
        # key); producer parses on receipt.
        if req_id is not None:
            wire_key = f"{req_id}{_KEY_SEP}{provider_string}"
        else:
            wire_key = provider_string

        # Allocate a private response tag for THIS pull and send ONE fixed-size,
        # self-identifying request. No lock: each mediator thread runs its own
        # pull concurrently. The per-pull tag keeps replies from colliding (gloo
        # routes by ``(peer, tag)``, never by content), and this thread's own
        # recv below is posted before the producer can reply on that tag (the
        # rendezvous requirement). The single atomic request can't interleave
        # with another consumer's on the shared TAG_REQUEST.
        response_tag = self._next_response_tag()
        dist.send(
            _encode_request(self._local_rank, response_tag, header_num_tokens, wire_key),
            group=group, group_dst=source_rank, tag=TAG_REQUEST,
        )

        if use_precomputed:
            return self._recv_precomputed(group, source_rank, meta, num_tokens, response_tag)
        else:
            dtype = meta.get("dtype", torch.float32) if isinstance(meta, dict) else (meta if isinstance(meta, torch.dtype) else torch.float32)
            return self._recv_legacy(
                group, source_rank, dtype, response_tag,
                module_path=module_path, meta=meta,
            )

    def _next_response_tag(self) -> int:
        """A distinct tag per in-flight pull (≥ ``TAG_RESPONSE_BASE``, never
        aliasing ``TAG_REQUEST``). ``itertools.count`` is atomic under the GIL,
        so concurrent mediator threads allocate without a lock."""
        return TAG_RESPONSE_BASE + (next(self._tag_counter) % _TAG_RANGE)

    def _recv_precomputed(self, group, source_rank, meta, num_tokens, tag):
        """Recv flat data using the learned module output shape — no shape
        on the wire.

        The producer's tensor is ``[num_tokens, *features]`` (vLLM's flat
        token-major layout). We keep the learned per-feature shape and
        substitute this request's ``num_tokens`` for the leading dim, since
        the token count is the only part that varies between requests.
        """
        dtype = meta["dtype"]
        num_outputs = meta["num_outputs"]
        module_shapes = meta["module_shapes"]

        shapes = []
        total_numel = 0
        for learned_shape in module_shapes:
            shape = (num_tokens, *learned_shape[1:])
            numel = 1
            for s in shape:
                numel *= s
            shapes.append((shape, numel))
            total_numel += numel

        flat = torch.zeros(total_numel, dtype=dtype)
        dist.recv(flat, group=group, group_src=source_rank, tag=tag)

        results = []
        offset = 0
        for shape, numel in shapes:
            results.append(flat[offset:offset + numel].reshape(shape).to(self._device))
            offset += numel

        if num_outputs == 1:
            return results[0]
        return tuple(results)

    def _recv_legacy(self, group, source_rank, dtype, tag,
                     module_path=None, meta=None):
        """Recv shape metadata then data (legacy fallback).

        Both messages ride the per-pull ``tag`` (FIFO on this one (peer, tag),
        single receiver), so the two-message response is safe under concurrency.
        Also learns the module's output shape from the wire so subsequent
        pulls of the same module use the precomputed path — see
        :meth:`_cache_module_shapes`.
        """
        shape_meta = torch.zeros(_META_SLOTS, dtype=torch.int64)
        dist.recv(shape_meta, group=group, group_src=source_rank, tag=tag)

        num_elements = int(shape_meta[0].item())
        shapes = []
        idx = 1
        total_numel = 0
        for _ in range(num_elements):
            ndim = int(shape_meta[idx].item())
            idx += 1
            shape = [int(shape_meta[idx + j].item()) for j in range(ndim)]
            idx += ndim
            numel = 1
            for s in shape:
                numel *= s
            shapes.append((shape, numel))
            total_numel += numel

        flat = torch.zeros(total_numel, dtype=dtype)
        dist.recv(flat, group=group, group_src=source_rank, tag=tag)

        results = []
        offset = 0
        for shape, numel in shapes:
            results.append(flat[offset:offset + numel].reshape(shape).to(self._device))
            offset += numel

        self._cache_module_shapes(module_path, meta, shapes, dtype)

        if num_elements == 1:
            return results[0]
        return tuple(results)

    @staticmethod
    def _should_use_precomputed(meta, num_tokens) -> bool:
        """Whether a pull can skip shape-on-wire (precomputed recv buffer).

        Requires this request's token count (to size the buffer) and a
        metadata entry whose ``module_shapes`` have been learned. Shapes
        start empty at init and are filled by :meth:`_cache_module_shapes`
        on the first legacy pull, so the first pull of each module is
        legacy and subsequent pulls are precomputed.
        """
        return bool(
            num_tokens > 0
            and isinstance(meta, dict)
            and meta.get("module_shapes")
        )

    def _cache_module_shapes(self, module_path, meta, shapes, dtype):
        """Learn a module's output shape(s) from a real legacy response.

        Records the per-element output shape so the next pull of this
        module takes the precomputed path. The token (leading) dimension
        is overridden per request at recv time, so what matters here is
        the trailing feature shape — we keep the whole observed shape and
        let :meth:`_recv_precomputed` substitute the live token count.
        """
        if module_path is None:
            return

        module_shapes = [tuple(shape) for shape, _ in shapes]

        # Mutate the resolved entry in place when it exists (keeps the
        # allgather-provided dtype and is agnostic to the prefix-tolerant
        # key under which it was found); otherwise create a fresh entry.
        if isinstance(meta, dict):
            entry = meta
        else:
            entry = {"dtype": dtype}
            self._meta_map[module_path] = entry
        entry["module_shapes"] = module_shapes
        entry["num_outputs"] = len(shapes)


def _provider_to_module_path(provider_string: str) -> str:
    """Strip '.output.iN' or '.input.iN' suffix to get the module path."""
    parts = provider_string.split(".")
    # Walk backwards to find and remove the access suffix
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("i") and parts[i][1:].isdigit():
            # Found iteration marker, the part before is "output" or "input"
            return ".".join(parts[: i - 1])
    return provider_string

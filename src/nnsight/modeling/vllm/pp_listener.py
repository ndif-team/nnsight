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

The producer's recv loop is a single thread but it NEVER blocks on serving: a
request whose value is already buffered is handed to a small reply pool, and a
request for a not-yet-produced value is PARKED (keyed by the awaited buffer key)
and served by ``dispatch_parked`` when the producer writes that value. This is
what stops one not-yet-ready pull from head-of-line-blocking every other rank's
request-``send`` at the gloo rendezvous (the multinode cross-stage deadlock).

Every reply is self-describing: a fixed-size header carrying the tensor count,
the value's true dtype, and per-tensor shapes, then the flat data — both on the
per-pull tag. Sizing always comes from the producer because the consumer cannot
predict the produced leading dim (token count) under run-ahead: the worker
builds its lazy placeholder before the matching forward is scheduled, so any
consumer-side token-count capture is unreliable (a wrong size either
under-allocates the recv buffer — gloo aborts — or over-allocates it — a
silently wrong-shaped tensor).
"""

from __future__ import annotations

import itertools
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .pp import PP_LISTENER_BACKOFF_S

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
# Reserved tag for the request-finalize drain barrier (``drain_barrier``), one
# above the entire response-tag range so it never aliases TAG_REQUEST or any
# in-flight reply. gloo routes strictly by ``(peer, tag)``, so the barrier's
# p2p on the listener thread's pull group is invisible to the serving recvs.
TAG_DRAIN = TAG_RESPONSE_BASE + _TAG_RANGE
_META_SLOTS = 32  # shape-header buffer size

# Error-reply sentinel in the shape header's slot 0 (a real reply always has
# >= 1 tensor there). When the producer can't serialize a requested value
# (a non-tensor like a dict-valued ``.inputs`` read, a mixed-dtype tuple, a
# shape too large for the header), it sends this header with the UTF-8 message
# length in slot 1 followed by the message bytes, so the blocked consumer raises
# instead of hanging. A per-op gloo recv timeout can't be the backstop instead:
# it closes the whole peer pair on expiry (probed), breaking every later pull.
_ERROR_SENTINEL = -1
_ERROR_MSG_CAP = 2048  # bound the on-wire error message

# Wire codec for the shape header's dtype slot (``shape_meta[1]``). The
# producer stamps the value's TRUE dtype here so the consumer sizes its recv
# buffer from the truth instead of guessing from the module's weight dtype —
# which is wrong for integer-valued outputs (e.g. sampled token ids are int32,
# not the model's bf16 compute dtype, so the guess under-sizes the buffer and
# gloo aborts on the size mismatch). Both PP ranks run identical code, so a
# fixed enum agrees on the wire. A dtype outside the table cannot be sized by
# the consumer; the producer error-replies it (``_encode_shape_header``
# raises) instead of stamping ``_DTYPE_CODE_UNKNOWN``.
_DTYPE_CODE_UNKNOWN = 0
_DTYPE_TO_CODE = {
    torch.float32: 1,
    torch.float64: 2,
    torch.float16: 3,
    torch.bfloat16: 4,
    torch.int64: 5,
    torch.int32: 6,
    torch.int16: 7,
    torch.int8: 8,
    torch.uint8: 9,
    torch.bool: 10,
    torch.complex64: 11,
    torch.complex128: 12,
}
# float8 variants exist only on recent torch builds.
for _code, _name in enumerate(
    ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"),
    start=13,
):
    if hasattr(torch, _name):
        _DTYPE_TO_CODE[getattr(torch, _name)] = _code
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}
# Bounded pool that performs reply sends off the recv loop. Replies are always
# quick (the consumer's recv is already posted), so a small pool keeps up; it
# only exists to keep the single recv loop free and to cap thread count (vs a
# thread per pull). Queued replies just wait for a free worker — they never
# block the recv loop or the producer.
_REPLY_POOL_SIZE = 32

# Separator between req_id and provider in the wire-encoded key. The wire key
# is ALWAYS ``"{req_id}|{provider}"`` (req_id empty for ``None``), decoding to
# the composite ``(provider, req_id)`` tuple the producer buffers under — the
# same key shape on both ends, no string-only fallback to diverge on.
_KEY_SEP = "|"

# A pull request is ONE fixed-size, self-identifying message on TAG_REQUEST:
# a two-message (header-then-key) request would interleave under concurrent
# senders and the producer would pair one consumer's header with another's key
# (size mismatch → gloo aborts the worker, or a garbled lookup key wedges it).
# Layout: 3 little-endian int64 [requester_rank, response_tag, key_len]
# followed by the UTF-8 key bytes, zero-padded to a constant size so the
# listener's single recv always matches the send size.
_REQUEST_HEADER = struct.Struct("<3q")
REQUEST_MSG_BYTES = 256


def _encode_request(requester_rank, response_tag, wire_key):
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
        buf, 0, requester_rank, response_tag, len(key_bytes)
    )
    buf[_REQUEST_HEADER.size:_REQUEST_HEADER.size + len(key_bytes)] = key_bytes
    # bytearray is writable, so frombuffer doesn't warn; clone to own the memory.
    return torch.frombuffer(buf, dtype=torch.uint8).clone()


def _decode_request(buf):
    """Inverse of :func:`_encode_request`.

    Returns ``(requester_rank, response_tag, wire_key)``.
    """
    raw = bytes(buf.numpy())
    requester_rank, response_tag, key_len = (
        _REQUEST_HEADER.unpack_from(raw, 0)
    )
    wire_key = raw[
        _REQUEST_HEADER.size:_REQUEST_HEADER.size + key_len
    ].decode("utf-8")
    return requester_rank, response_tag, wire_key


class PPListener:
    """Cross-rank tensor pull service.

    Producer (background listener thread): recvs fixed-size requests on
    TAG_REQUEST, sends each reply on the per-pull tag carried in its request.

    Consumer (mediator thread): allocates a per-pull tag, sends one fixed-size
    request on TAG_REQUEST, recvs its reply on that tag. Many mediator threads
    may do this concurrently.

    The request packs ``[requester_rank, response_tag, key_len] + key`` (see
    ``_encode_request``); the reply is shape metadata then flat data, both on
    the per-pull tag.

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
    ):
        self._buffer = buffer
        self._condition = condition
        self._pull_group = pull_group
        self._local_rank = local_rank
        self._device = device
        self._thread: Optional[threading.Thread] = None
        # Non-blocking serve. A request whose value isn't buffered yet is PARKED
        # here (key -> list of pending requests) instead of blocking the recv
        # loop; ``dispatch_parked`` serves it when the producer writes the value.
        # Accessed only under ``_condition`` (the buffer lock), so check-and-park
        # in the recv loop races safely against the producer's write+dispatch.
        self._parked: Dict[Any, list] = {}
        # Bounded pool that runs the (always-quick) reply sends off the recv loop.
        self._reply_pool = ThreadPoolExecutor(
            max_workers=_REPLY_POOL_SIZE, thread_name_prefix="pp-reply"
        )
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
        """Drop buffer entries and error-reply pulls they would have served.

        Args:
            req_ids: Optional iterable of request ids.  When supplied,
                only keys whose 2-tuple tail equals one of these ids
                are removed — in-flight requests' entries stay intact.
                When ``None``, falls back to the legacy blanket clear
                (all keys removed).

        The clear must stay SCOPED at request completion: with composite
        keys ``(provider, req_id)``, a blanket clear at ANY request's
        completion would also wipe concurrent in-flight requests'
        slices and strand their still-pending cross-rank pulls.
        """
        abandoned = []  # parked requests whose value will never be produced
        with self._condition:
            if req_ids is None:
                self._buffer.clear()
                # A blanket clear means the request set is done; any still-parked
                # pull will never be served now — collect them to error-reply
                # below (a silent drop leaves the blocked consumer hung).
                for reqs in self._parked.values():
                    abandoned.extend(reqs)
                self._parked.clear()
            else:
                id_set = set(req_ids)
                to_remove = [
                    k for k in self._buffer
                    if isinstance(k, tuple) and len(k) == 2 and k[1] in id_set
                ]
                for k in to_remove:
                    del self._buffer[k]
                # Pulls still parked for a finished request: their value will
                # never be produced (commonly a run-ahead worker that pulled one
                # generation step past the end). Error-reply each instead of a
                # silent drop, so its blocked consumer raises and its worker
                # thread exits — otherwise it leaks (a per-op gloo recv timeout
                # can't rescue it; it closes the whole peer pair).
                for k in [
                    k for k in self._parked
                    if isinstance(k, tuple) and len(k) == 2 and k[1] in id_set
                ]:
                    abandoned.extend(self._parked.pop(k))
            self._condition.notify_all()

        for req in abandoned:
            self._reply_pool.submit(
                self._serve_error_reply, req,
                "value was never produced (request finalized before this "
                "cross-stage pull resolved)",
            )

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
        self._reply_pool.shutdown(wait=False)

    def _listen_loop(self):
        group = self._pull_group
        world_size = dist.get_world_size(group)
        other_ranks = [r for r in range(world_size) if r != self._local_rank]

        while not self._stop_event.is_set():
            try:
                # Recv ONE fixed-size, self-identifying request on TAG_REQUEST.
                # A single atomic message (vs. the old header+key pair) is what
                # makes concurrent senders safe. ``group_src`` stays wildcard for
                # PP>2; the request carries ``requesting_rank`` for the reply.
                #
                # The recv loop must NEVER block on serving: if the requested
                # value isn't buffered yet, we PARK the request (served later by
                # ``dispatch_parked``) instead of waiting. A blocking serve here
                # would stop the loop from posting the next ``recv``, freezing
                # every other rank's request-``send`` at the rendezvous — the
                # multinode cross-stage head-of-line deadlock.
                req_buf = torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8)
                src = other_ranks[0] if len(other_ranks) == 1 else None
                dist.recv(req_buf, group=group, group_src=src, tag=TAG_REQUEST)

                requesting_rank, response_tag, encoded = (
                    _decode_request(req_buf)
                )

                # Wire key is ``"{req_id}|{provider}"``; decode to the same
                # composite ``(provider, req_id)`` the producer buffers under.
                req_id_str, provider_string = encoded.split(_KEY_SEP, 1)
                lookup_key = (provider_string, req_id_str or None)

                req = (requesting_rank, response_tag)
                # Check-and-park under the buffer lock so it races safely with
                # the producer's write + ``dispatch_parked``: either we already
                # see the value (serve now via the pool) or we park and the
                # producer serves us when it writes. Exactly one path fires, so
                # no request is dropped or double-served.
                with self._condition:
                    if lookup_key in self._buffer:
                        value = self._buffer[lookup_key]
                        self._reply_pool.submit(self._serve_reply, req, value)
                    else:
                        self._parked.setdefault(lookup_key, []).append(req)

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
                if self._stop_event.wait(timeout=PP_LISTENER_BACKOFF_S):
                    return

    def dispatch_parked(self, key, value):
        """Serve any pulls parked waiting for ``key``.

        Called by the producer (:meth:`Mediator.handle_value_event`) right after
        it writes ``value`` into the buffer under ``pp_buffer_condition`` (the
        same object as ``_condition``). Pops the parked requests for ``key`` and
        hands each to the reply pool. The condition is a reentrant lock, so this
        is safe to call with it already held.
        """
        with self._condition:
            waiters = self._parked.pop(key, None)
        if not waiters:
            return
        for req in waiters:
            self._reply_pool.submit(self._serve_reply, req, value)

    @staticmethod
    def _encode_shape_header(cpu_tensors):
        """Build the reply's shape header: slot 0 = tensor count, slot 1 =
        dtype code, then ``[ndim, *dims]`` per tensor. Raises ``ValueError``
        if the value needs more slots than the fixed buffer holds, or if the
        dtype is outside the wire codec — either way caught by
        ``_serve_reply`` and turned into an error reply, not a wedged consumer.
        """
        needed = 2 + sum(1 + t.ndim for t in cpu_tensors)
        if needed > _META_SLOTS:
            raise ValueError(
                f"cross-stage value needs {needed} shape-header slots > "
                f"{_META_SLOTS} ({len(cpu_tensors)} tensors, shapes "
                f"{[tuple(t.shape) for t in cpu_tensors]})"
            )
        shape_meta = torch.zeros(_META_SLOTS, dtype=torch.int64)
        shape_meta[0] = len(cpu_tensors)
        # Slot 1 carries the value's real dtype so the consumer sizes its recv
        # buffer from the truth, not a weight-derived guess. All ``cpu_tensors``
        # share one dtype (the ``cat`` in ``_serve_reply`` requires it).
        if cpu_tensors:
            code = _DTYPE_TO_CODE.get(cpu_tensors[0].dtype, _DTYPE_CODE_UNKNOWN)
            if code == _DTYPE_CODE_UNKNOWN:
                raise ValueError(
                    f"cross-stage value dtype {cpu_tensors[0].dtype} is not "
                    f"wire-encodable (no code in the dtype codec)"
                )
            shape_meta[1] = code
        idx = 2
        for t in cpu_tensors:
            shape_meta[idx] = t.ndim
            idx += 1
            for s in t.shape:
                shape_meta[idx] = s
                idx += 1
        return shape_meta

    def _serve_error_reply(self, req, message):
        """Tell a blocked consumer its pull failed, instead of leaving it hung.

        Rides the normal reply channel: a header with slot 0 ==
        ``_ERROR_SENTINEL`` and the UTF-8 message length in slot 1, followed by
        the message bytes — both on the pull's private response tag.
        """
        requesting_rank, response_tag = req
        group = self._pull_group
        msg = message.encode("utf-8")[:_ERROR_MSG_CAP]
        header = torch.zeros(_META_SLOTS, dtype=torch.int64)
        header[0] = _ERROR_SENTINEL
        header[1] = len(msg)
        try:
            dist.send(header, group=group, group_dst=requesting_rank, tag=response_tag)
            dist.send(
                torch.frombuffer(bytearray(msg), dtype=torch.uint8).clone(),
                group=group, group_dst=requesting_rank, tag=response_tag,
            )
        except Exception:
            if not dist.is_initialized() or self._stop_event.is_set():
                return
            import traceback
            traceback.print_exc()

    def _serve_reply(self, req, value):
        """Send one reply on its per-pull response tag (runs on the reply pool):
        shape metadata then flat data. The consumer's recv on this tag is
        already posted, so each ``send`` completes promptly.

        The whole reply is PREPARED before any send, so a serialization failure
        (a non-tensor value, a mixed-dtype tuple, a shape too big for the header)
        is caught while we can still send an error reply — never after a partial
        send that would desync the consumer's recv (and leave it hung, since a
        gloo recv timeout can't safely rescue it — see ``_serve_error_reply``).
        """
        requesting_rank, response_tag = req
        group = self._pull_group

        try:
            # Normalize to list of tensors (handles both tensor and tuple).
            tensors = list(value) if isinstance(value, (tuple, list)) else [value]
            cpu_tensors = [t.detach().contiguous().cpu() for t in tensors]
            # ``cat`` validates same-dtype/numeric up front (a mixed-dtype tuple
            # raises here, before any send).
            flat = torch.cat([t.contiguous().view(-1) for t in cpu_tensors])
            shape_meta = self._encode_shape_header(cpu_tensors)
        except Exception as exc:
            self._serve_error_reply(req, f"{type(exc).__name__}: {exc}")
            return

        try:
            # On the per-pull response tag carried in the request, so concurrent
            # consumers each receive only their own reply.
            dist.send(shape_meta, group=group, group_dst=requesting_rank, tag=response_tag)
            dist.send(flat, group=group, group_dst=requesting_rank, tag=response_tag)
        except Exception:
            if not dist.is_initialized() or self._stop_event.is_set():
                return
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Consumer: pull tensor from remote rank
    # ------------------------------------------------------------------

    def pull_from_remote(
        self,
        source_rank: int,
        provider_string: str,
        req_id: Optional[str] = None,
    ):
        """Pull tensor(s) from a remote rank.

        Args:
            source_rank: PP rank that owns the module.
            provider_string: Module-level provider (``"…output.iN"``).
            req_id: vLLM request id. The wire key encodes
                ``"{req_id}|{provider}"`` and the producer looks up the
                composite ``(provider, req_id)`` tuple so concurrent
                requests reading the same provider on the same forward
                can't deliver each other's slices. ``None`` encodes as an
                empty id, matching a producer-side key of
                ``(provider, None)``.
        """
        if self._pull_group is None:
            raise RuntimeError("No pull_group configured for cross-rank pull")

        group = self._pull_group

        # Wire-encode the composite key; producer parses on receipt.
        wire_key = f"{req_id if req_id is not None else ''}{_KEY_SEP}{provider_string}"

        # Allocate a private response tag for THIS pull and send ONE fixed-size,
        # self-identifying request. No lock: each mediator thread runs its own
        # pull concurrently. The per-pull tag keeps replies from colliding (gloo
        # routes by ``(peer, tag)``, never by content), and this thread's own
        # recv below is posted before the producer can reply on that tag (the
        # rendezvous requirement). The single atomic request can't interleave
        # with another consumer's on the shared TAG_REQUEST.
        response_tag = self._next_response_tag()
        dist.send(
            _encode_request(self._local_rank, response_tag, wire_key),
            group=group, group_dst=source_rank, tag=TAG_REQUEST,
        )

        return self._recv_reply(
            group, source_rank, response_tag,
            module_path=_provider_to_module_path(provider_string),
        )

    def drain_barrier(self):
        """Cross-PP-rank barrier on the reserved ``TAG_DRAIN`` tag.

        Used at request finalize (``GPUModelRunner.collect_nnsight``) so NO rank
        tears down its ``pp_hook_buffer`` until EVERY rank's mediator workers have
        drained their cross-stage pulls. Each rank arrives here only AFTER joining
        its own worker, so reaching the barrier means "my worker is fully drained";
        and each rank keeps SERVING peers on its listener thread (the buffer is
        still alive — clear happens after this returns), so a peer's in-flight
        pulls are satisfied from the live buffer instead of blocking on a torn-down
        peer until the 5 s join timeout (the PB1 stall).

        Safe to call collectively: ``collect_nnsight`` runs via ``collective_rpc``
        with identical ``finished_req_ids`` on every rank, and only TP-rank-0 of
        each PP stage reaches it — exactly the members of this pull group. The
        pairwise exchange is rank-ordered (lower sends first) so it cannot
        deadlock, and runs on ``TAG_DRAIN`` so it never collides with the
        listener's ``TAG_REQUEST`` recvs or in-flight reply tags. No-op without a
        peer (PP disabled / single rank).
        """
        group = self._pull_group
        if group is None:
            return
        world = dist.get_world_size(group)
        if world < 2:
            return
        me = self._local_rank
        token = torch.zeros(1, dtype=torch.uint8)
        for r in range(world):
            if r == me:
                continue
            if me < r:
                dist.send(token, group=group, group_dst=r, tag=TAG_DRAIN)
                dist.recv(token, group=group, group_src=r, tag=TAG_DRAIN)
            else:
                dist.recv(token, group=group, group_src=r, tag=TAG_DRAIN)
                dist.send(token, group=group, group_dst=r, tag=TAG_DRAIN)

    def _next_response_tag(self) -> int:
        """A distinct tag per in-flight pull (≥ ``TAG_RESPONSE_BASE``, never
        aliasing ``TAG_REQUEST``). ``itertools.count`` is atomic under the GIL,
        so concurrent mediator threads allocate without a lock."""
        return TAG_RESPONSE_BASE + (next(self._tag_counter) % _TAG_RANGE)

    def _recv_reply(self, group, source_rank, tag, module_path=None):
        """Recv one reply: shape metadata then data.

        Both messages ride the per-pull ``tag`` (FIFO on this one (peer, tag),
        single receiver), so the two-message response is safe under
        concurrency. The recv buffer is sized entirely from the header — the
        producer is the only side that knows the value's true shape and dtype
        under run-ahead.
        """
        shape_meta = torch.zeros(_META_SLOTS, dtype=torch.int64)
        dist.recv(shape_meta, group=group, group_src=source_rank, tag=tag)

        num_elements = int(shape_meta[0].item())
        if num_elements == _ERROR_SENTINEL:
            # The producer couldn't serialize this value; slot 1 is the message
            # length, the next message is the UTF-8 message. Raise instead of
            # hanging (a serve failure used to just print on the producer and
            # strand this recv forever).
            err_len = int(shape_meta[1].item())
            err_buf = torch.zeros(err_len, dtype=torch.uint8)
            dist.recv(err_buf, group=group, group_src=source_rank, tag=tag)
            msg = bytes(err_buf.numpy()).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"PP cross-stage pull of {module_path!r} failed on its owning "
                f"rank ({source_rank}): {msg}"
            )
        # The producer stamped the value's real dtype in slot 1 (it
        # error-replies dtypes outside the codec, so an unknown code here is a
        # protocol bug, not a servable value).
        recv_dtype = _CODE_TO_DTYPE.get(int(shape_meta[1].item()))
        if recv_dtype is None:
            raise RuntimeError(
                f"PP cross-stage pull of {module_path!r}: reply header carries "
                f"unknown dtype code {int(shape_meta[1].item())}"
            )
        shapes = []
        idx = 2
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

        flat = torch.zeros(total_numel, dtype=recv_dtype)
        dist.recv(flat, group=group, group_src=source_rank, tag=tag)

        results = []
        offset = 0
        for shape, numel in shapes:
            results.append(flat[offset:offset + numel].reshape(shape).to(self._device))
            offset += numel

        if num_elements == 1:
            return results[0]
        return tuple(results)


def _provider_to_module_path(provider_string: str) -> str:
    """Strip the access suffix (``.output.iN`` / ``.input.iN`` / ``.iN``) to
    get the module path.

    The part before the iteration marker is only dropped when it actually is
    the eproperty name (``output``/``input``/``inputs``). A root eproperty's
    provider is ``model.logits.iN`` — its trailing part IS the module name;
    unconditionally dropping it collapses both ``model.logits`` and
    ``model.samples`` to ``"model"``. Used to name the module in pull error
    messages; ``pp_envoy`` mirrors this canonicalization for its dtype-hint
    lookup.
    """
    parts = provider_string.split(".")
    # Walk backwards to find and remove the access suffix
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("i") and parts[i][1:].isdigit():
            if i > 0 and parts[i - 1] in ("output", "input", "inputs"):
                return ".".join(parts[: i - 1])
            return ".".join(parts[:i])
    return provider_string

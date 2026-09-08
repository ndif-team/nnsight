"""Cross-rank tensor pulls for pipeline parallelism, over a dedicated gloo group.

Two sides, asymmetric by design:

**Producer** (every rank, for the modules it owns): a background listener thread
recvs fixed-size pull requests on the shared ``TAG_REQUEST`` and replies on the
per-pull tag each request carries. Values are published into a buffer as the
forward reaches them; a request for a not-yet-produced value is PARKED (keyed by
the awaited buffer key) and served by :meth:`PPListener.dispatch_parked` when
the producer writes that value — the recv loop itself never blocks on serving,
which is what stops one not-yet-ready pull from head-of-line-blocking every
other rank's request-``send`` at the gloo rendezvous. The listener thread is
legal under the greenlet engine because it never touches a greenlet: it only
reads buffered values and sends bytes.

**Consumer** (the rank whose intervention block consumes a remote-owned value):
split-phase, never blocking the forward thread. :meth:`PPListener.begin_pull`
sends the request — at intercept time, mid-forward — and hands the reply's recv
sequence to a small waiter pool; it returns a :class:`Pull` whose
:attr:`~Pull.ready` flips when the value has fully arrived, and whose
:meth:`~Pull.complete` collects it at a serve point. The waiter thread is a
dumb byte-mover (recv header, recv data, assemble on CPU) that never touches a
greenlet; its blocking recv is what lets the transfer complete while this
rank's forward still runs. gloo is a rendezvous transport, so issuing the
request and posting the reply recv early is not an optimization — it is what
lets the producer's send complete before this rank's serve point. (gloo's
``irecv`` handles cannot replace the waiter: probed on this build, their
``is_completed()`` never flips outside ``wait()``, and a ``wait(timeout)``
expiry closes the whole peer pair.)

Wire rules (gloo routes point-to-point traffic strictly by ``(peer, tag)``,
FIFO per pair, and aborts on a size-mismatched recv):

  - **Requests** are ONE fixed-size, self-identifying message on the shared
    ``TAG_REQUEST`` — a single atomic message cannot interleave with another
    consumer's request. It carries the requester rank, a per-pull response tag,
    and the lookup key.
  - **Replies** ride the per-pull response tag carried in the request, so
    concurrent consumers each receive only their own reply. Every reply is
    self-describing: a fixed-size header carrying the tensor count, the value's
    TRUE dtype, and per-tensor shapes, then the flat data. Sizing always comes
    from the producer: only it knows the produced shape and dtype (sampled ids
    are int32, not the model's compute dtype — a weight-derived guess
    under-sizes the buffer and gloo aborts).
  - **Errors** ride the same channel: a header whose first slot is the error
    sentinel, then the message bytes, so a consumer raises instead of hanging.
    A per-op gloo recv timeout cannot be the backstop instead: expiry closes
    the whole peer pair (probed), breaking every later pull.
"""

from __future__ import annotations

import itertools
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .pp import PP_LISTENER_BACKOFF_S, PP_PULL_TIMEOUT_S

# Requests all arrive on this one well-known tag (the listener can only
# pre-post a recv on a tag it knows). Responses ride a per-pull tag carried
# IN the request, starting at ``TAG_RESPONSE_BASE`` so they never alias
# ``TAG_REQUEST``.
TAG_REQUEST = 0
TAG_RESPONSE_BASE = 1024
# Response tags cycle through this range; with short-lived pulls a tag is only
# reused after a full cycle, never while one is concurrently in flight.
_TAG_RANGE = 1 << 20
# Reserved tag for the request-finalize drain barrier, one above the entire
# response-tag range so it never aliases TAG_REQUEST or any in-flight reply.
TAG_DRAIN = TAG_RESPONSE_BASE + _TAG_RANGE
_META_SLOTS = 32  # shape-header buffer size

# Error-reply sentinel in the shape header's slot 0 (a real reply always has
# >= 1 tensor there); slot 1 then carries the UTF-8 message length.
_ERROR_SENTINEL = -1
_ERROR_MSG_CAP = 2048  # bound the on-wire error message

# Wire codec for the shape header's dtype slot. Both ranks run identical code,
# so a fixed enum agrees on the wire. A dtype outside the table cannot be sized
# by the consumer; the producer error-replies it instead.
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
# quick (the consumer's recv is already posted), so a small pool keeps up.
_REPLY_POOL_SIZE = 32

# Separator between req_id and provider in the wire-encoded key. The wire key
# is ALWAYS ``"{req_id}|{provider}"`` (req_id empty for ``None``), decoding to
# the composite ``(provider, req_id)`` tuple the producer buffers under.
_KEY_SEP = "|"

# A pull request is ONE fixed-size, self-identifying message on TAG_REQUEST:
# 3 little-endian int64 [requester_rank, response_tag, key_len] followed by the
# UTF-8 key bytes, zero-padded to a constant size so the listener's single recv
# always matches the send size.
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
    _REQUEST_HEADER.pack_into(buf, 0, requester_rank, response_tag, len(key_bytes))
    buf[_REQUEST_HEADER.size:_REQUEST_HEADER.size + len(key_bytes)] = key_bytes
    # bytearray is writable, so frombuffer doesn't warn; clone to own the memory.
    return torch.frombuffer(buf, dtype=torch.uint8).clone()


def _decode_request(buf):
    """Inverse of :func:`_encode_request` → ``(requester_rank, response_tag, wire_key)``."""
    raw = bytes(buf.numpy())
    requester_rank, response_tag, key_len = _REQUEST_HEADER.unpack_from(raw, 0)
    wire_key = raw[
        _REQUEST_HEADER.size:_REQUEST_HEADER.size + key_len
    ].decode("utf-8")
    return requester_rank, response_tag, wire_key


def provider_to_module_path(provider_string: str) -> str:
    """Strip the access suffix (``.output.iN`` / ``.input.iN`` / ``.iN``) to
    get the module path.

    The part before the iteration marker is only dropped when it actually is
    the eproperty name (``output``/``input``/``inputs``). A root eproperty's
    provider is ``model.logits.iN`` — its trailing part IS the module name;
    unconditionally dropping it collapses both ``model.logits`` and
    ``model.samples`` to ``"model"``. Used to name the module in pull error
    messages.
    """
    parts = provider_string.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("i") and parts[i][1:].isdigit():
            if i > 0 and parts[i - 1] in ("output", "input", "inputs"):
                return ".".join(parts[: i - 1])
            return ".".join(parts[:i])
    return provider_string


class Pull:
    """One in-flight cross-stage pull, received on a waiter thread.

    Built by :meth:`PPListener.begin_pull`, which has already sent the request
    and handed the reply's recv sequence to the waiter pool — the issue-early
    half of the overlap contract. The waiter blocks in the recvs (header, then
    data sized and typed from it), assembles the value on CPU, and flips
    :attr:`ready`; the transfer therefore completes while the forward runs,
    without the forward thread ever blocking.

    :meth:`complete` collects the value at a serve point: wait until ready
    (raising loudly after ``timeout`` — the deadline is a fatal diagnostic;
    expiry means the owning rank never produced or served the value), move it
    to the target device on the calling thread, and re-raise on this thread any
    error the pull carried (the producer's error reply, or a protocol error on
    the waiter), so the consuming worker fails at the line that forced the
    value.
    """

    def __init__(
        self,
        listener: "PPListener",
        source_rank: int,
        provider_string: str,
        tag: int,
    ) -> None:
        self._listener = listener
        self._source_rank = source_rank
        self._provider = provider_string
        self._module_path = provider_to_module_path(provider_string)
        self._tag = tag
        # Flipped by the waiter once the value (or an error) has fully arrived.
        self._done = threading.Event()
        # Exactly one of these is set when _done flips.
        self._value: Any = None
        self._error: Optional[BaseException] = None

    @property
    def ready(self) -> bool:
        """Whether the value (or its error) has fully arrived."""
        return self._done.is_set()

    def complete(self, timeout: float = PP_PULL_TIMEOUT_S) -> Any:
        """Wait for the pull and return its value (on the listener's device)."""
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"PP cross-stage pull of {self._module_path!r} from rank "
                f"{self._source_rank} did not complete within {timeout}s. "
                f"The owning rank never produced or served this value — check "
                f"that its forward reaches the module and that the request is "
                f"still live there."
            )
        if self._error is not None:
            raise self._error
        # Device placement happens here, on the collecting thread, so the copy
        # is ordered on that thread's stream rather than the waiter's.
        device = self._listener._device
        if isinstance(self._value, tuple):
            return tuple(t.to(device) for t in self._value)
        return self._value.to(device)

    def _receive(self) -> None:
        """Recv the full reply (runs on the waiter pool; blocking is its job)."""
        try:
            group = self._listener._pull_group
            header = torch.zeros(_META_SLOTS, dtype=torch.int64)
            dist.recv(header, group=group, group_src=self._source_rank, tag=self._tag)

            first = int(header[0].item())
            if first == _ERROR_SENTINEL:
                err_buf = torch.zeros(int(header[1].item()), dtype=torch.uint8)
                dist.recv(
                    err_buf, group=group, group_src=self._source_rank, tag=self._tag
                )
                msg = bytes(err_buf.numpy()).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"PP cross-stage pull of {self._module_path!r} failed on "
                    f"its owning rank ({self._source_rank}): {msg}"
                )

            # A real reply: dtype code in slot 1, then [ndim, *dims] per tensor.
            # The data buffer is sized and typed entirely from the header — the
            # producer is the only side that knows the produced shape under
            # run-ahead.
            recv_dtype = _CODE_TO_DTYPE.get(int(header[1].item()))
            if recv_dtype is None:
                raise RuntimeError(
                    f"PP cross-stage pull of {self._module_path!r}: reply "
                    f"header carries unknown dtype code {int(header[1].item())}"
                )
            shapes = []
            idx = 2
            total_numel = 0
            for _ in range(first):
                ndim = int(header[idx].item())
                idx += 1
                shape = [int(header[idx + j].item()) for j in range(ndim)]
                idx += ndim
                numel = 1
                for s in shape:
                    numel *= s
                shapes.append((shape, numel))
                total_numel += numel

            flat = torch.zeros(total_numel, dtype=recv_dtype)
            dist.recv(flat, group=group, group_src=self._source_rank, tag=self._tag)

            results = []
            offset = 0
            for shape, numel in shapes:
                results.append(flat[offset:offset + numel].reshape(shape))
                offset += numel
            self._value = results[0] if len(results) == 1 else tuple(results)
        except BaseException as error:
            self._error = error
        finally:
            self._done.set()


class PPListener:
    """Cross-rank tensor pull service for one rank.

    Producer: :meth:`start` launches the background listener thread; values are
    published into ``buffer`` (under ``condition``) by the interleaver as the
    forward reaches them, and :meth:`dispatch_parked` serves any pulls parked
    on a just-published key.

    Consumer: :meth:`begin_pull` issues a split-phase :class:`Pull`.

    Args:
        buffer: The rank's published-value buffer, keyed ``(provider, req_id)``.
        condition: The lock/condition guarding ``buffer`` — shared with the
            interleaver that publishes into it.
        pull_group: The dedicated gloo group pulls ride on (``None`` disables).
        local_rank: This rank's index within ``pull_group``.
        device: Where pulled values are materialized.
    """

    def __init__(
        self,
        buffer: Dict[Any, Any],
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
        # Non-blocking serve: requests for not-yet-produced values, key -> list
        # of pending (rank, tag) requests. Accessed only under ``_condition``,
        # so check-and-park in the recv loop races safely against the
        # producer's write+dispatch.
        self._parked: Dict[Any, list] = {}
        self._reply_pool = ThreadPoolExecutor(
            max_workers=_REPLY_POOL_SIZE, thread_name_prefix="pp-reply"
        )
        # Waiter pool for this rank's own outgoing pulls (consumer side): each
        # job is one Pull's blocking recv sequence. Bounded like the reply
        # pool; a 33rd concurrent pull queues, and its reply is simply held at
        # the rendezvous until a slot frees — progress is guaranteed because
        # earlier pulls complete when their values publish (or error-reply at
        # finalize), independent of this pool.
        self._pull_pool = ThreadPoolExecutor(
            max_workers=_REPLY_POOL_SIZE, thread_name_prefix="pp-pull"
        )
        # Set by ``stop()`` to break the listen loop cleanly; also checked
        # after every caught exception so a torn-down ``dist`` context doesn't
        # busy-loop the listener at 100% CPU.
        self._stop_event = threading.Event()
        # Distinct response tag per in-flight pull. All consumer-side calls run
        # on the forward thread (workers are greenlets on it), so allocation is
        # single-threaded; itertools.count is atomic under the GIL regardless.
        self._tag_counter = itertools.count()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def clear_buffer(self, req_ids=None):
        """Drop buffer entries and error-reply pulls they would have served.

        Args:
            req_ids: Optional iterable of request ids. When supplied, only keys
                whose 2-tuple tail equals one of these ids are removed —
                in-flight requests' entries stay intact. ``None`` clears all.

        The clear must stay SCOPED at request completion: with composite keys
        ``(provider, req_id)``, a blanket clear at ANY request's completion
        would also wipe concurrent in-flight requests' slices and strand their
        still-pending cross-rank pulls. Pulls parked for a cleared request will
        never be served now (commonly a run-ahead worker that pulled one
        generation step past the end); each gets an error reply instead of a
        silent drop, so its blocked consumer raises rather than leaking.
        """
        abandoned = []
        with self._condition:
            if req_ids is None:
                self._buffer.clear()
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
        self._pull_pool.shutdown(wait=False)

    def _listen_loop(self):
        group = self._pull_group
        world_size = dist.get_world_size(group)
        other_ranks = [r for r in range(world_size) if r != self._local_rank]

        while not self._stop_event.is_set():
            try:
                # Recv ONE fixed-size, self-identifying request on TAG_REQUEST.
                # ``group_src`` stays wildcard for PP>2; the request carries the
                # requester's rank for the reply. The loop must NEVER block on
                # serving: a not-yet-buffered value parks the request instead,
                # keeping the next recv posted so other ranks' request-sends
                # never freeze at the rendezvous.
                req_buf = torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8)
                src = other_ranks[0] if len(other_ranks) == 1 else None
                dist.recv(req_buf, group=group, group_src=src, tag=TAG_REQUEST)

                requesting_rank, response_tag, encoded = _decode_request(req_buf)

                # Wire key is ``"{req_id}|{provider}"``; decode to the same
                # composite ``(provider, req_id)`` the producer buffers under.
                req_id_str, provider_string = encoded.split(_KEY_SEP, 1)
                lookup_key = (provider_string, req_id_str or None)

                req = (requesting_rank, response_tag)
                # Check-and-park under the buffer lock so it races safely with
                # the producer's write + ``dispatch_parked``: either we already
                # see the value (serve now via the pool) or we park and the
                # producer serves us when it writes. Exactly one path fires.
                with self._condition:
                    if lookup_key in self._buffer:
                        value = self._buffer[lookup_key]
                        self._reply_pool.submit(self._serve_reply, req, value)
                    else:
                        self._parked.setdefault(lookup_key, []).append(req)

            except Exception:
                # A torn-down dist context (worker shutdown, peer crash) makes
                # every recv raise instantly; exit instead of spinning. Genuine
                # transient errors log and retry after a short backoff.
                if not dist.is_initialized() or self._stop_event.is_set():
                    return
                import traceback
                traceback.print_exc()
                if self._stop_event.wait(timeout=PP_LISTENER_BACKOFF_S):
                    return

    def dispatch_parked(self, key, value):
        """Serve any pulls parked waiting for ``key``.

        Called by the publisher right after it writes ``value`` into the buffer
        under the shared condition (reentrant, so safe to call with it held).
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
        dtype code, then ``[ndim, *dims]`` per tensor. Raises ``ValueError`` on
        a value that doesn't fit the header or the dtype codec — caught by
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
        # buffer from the truth, not a weight-derived guess. All tensors share
        # one dtype (the ``cat`` in ``_serve_reply`` requires it).
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
        already posted (issue-early), so each ``send`` completes promptly.

        The whole reply is PREPARED before any send, so a serialization failure
        (a non-tensor value, a mixed-dtype tuple, a shape too big for the
        header) is caught while an error reply can still be sent — never after
        a partial send that would desync the consumer's posted recvs.
        """
        requesting_rank, response_tag = req
        group = self._pull_group

        try:
            tensors = list(value) if isinstance(value, (tuple, list)) else [value]
            cpu_tensors = [t.detach().contiguous().cpu() for t in tensors]
            # The data message is one flat tensor and the header carries one
            # dtype (slot 1), so every element must share it. ``cat`` cannot
            # enforce this: it silently promotes a mixed input, and the
            # consumer would rebuild every element in the first tensor's
            # dtype. A mixed container is real user-reachable data (a remote
            # layer's ``.inputs`` bundles int64 positions with bf16 hidden
            # states), so refuse it explicitly and let the error reply name
            # the location. Supporting mixed containers instead would take a
            # dtype code per tensor in the header plus a single uint8
            # byte-blob data message (each tensor viewed as bytes).
            dtypes = {t.dtype for t in cpu_tensors}
            if len(dtypes) > 1:
                raise ValueError(
                    f"cross-stage value mixes dtypes "
                    f"{sorted(str(d) for d in dtypes)}; a reply ships one "
                    f"dtype, so read the elements separately"
                )
            flat = torch.cat([t.contiguous().view(-1) for t in cpu_tensors])
            shape_meta = self._encode_shape_header(cpu_tensors)
        except Exception as exc:
            self._serve_error_reply(req, f"{type(exc).__name__}: {exc}")
            return

        try:
            dist.send(shape_meta, group=group, group_dst=requesting_rank, tag=response_tag)
            dist.send(flat, group=group, group_dst=requesting_rank, tag=response_tag)
        except Exception:
            if not dist.is_initialized() or self._stop_event.is_set():
                return
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Consumer: split-phase pulls
    # ------------------------------------------------------------------

    def begin_pull(
        self,
        source_rank: int,
        provider_string: str,
        req_id: Optional[str] = None,
    ) -> Pull:
        """Issue a cross-stage pull; return its :class:`Pull` to pump and complete.

        Sends the fixed-size request and posts the header ``irecv`` before
        returning — the issue-early half of the overlap contract, called at
        intercept time mid-forward. The request send completes as soon as the
        owning rank's listen loop takes it (that loop always has the next recv
        posted and never blocks on serving).

        Args:
            source_rank: PP rank that owns the module.
            provider_string: Module-level provider (``"…output.iN"``).
            req_id: vLLM request id. The wire key encodes
                ``"{req_id}|{provider}"`` and the producer looks up the
                composite ``(provider, req_id)`` so concurrent requests reading
                the same provider on the same forward can't deliver each
                other's slices.
        """
        if self._pull_group is None:
            raise RuntimeError("No pull_group configured for cross-rank pull")

        group = self._pull_group
        wire_key = f"{req_id if req_id is not None else ''}{_KEY_SEP}{provider_string}"
        response_tag = self._next_response_tag()

        # The waiter is submitted before the request goes out; the rendezvous
        # holds the producer's reply until the recv is actually posted, so the
        # order is about keeping the reply path independent of pool scheduling,
        # not correctness.
        pull = Pull(self, source_rank, provider_string, response_tag)
        self._pull_pool.submit(pull._receive)

        dist.send(
            _encode_request(self._local_rank, response_tag, wire_key),
            group=group, group_dst=source_rank, tag=TAG_REQUEST,
        )
        return pull

    def drain_barrier(self):
        """Cross-PP-rank barrier on the reserved ``TAG_DRAIN`` tag.

        Used at request finalize so NO rank tears down its published-value
        buffer until EVERY rank has completed its workers' cross-stage pulls.
        Each rank arrives here only after its own serve point has drained, and
        keeps SERVING peers on its listener thread throughout (the buffer is
        still alive — clear happens after this returns), so a peer's in-flight
        pulls are satisfied from the live buffer instead of stalling against a
        torn-down peer.

        The pairwise exchange is rank-ordered (lower sends first) so it cannot
        deadlock, and rides ``TAG_DRAIN`` so it never collides with the
        listener's ``TAG_REQUEST`` recvs or in-flight reply tags. No-op without
        a peer.
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
        aliasing ``TAG_REQUEST``)."""
        return TAG_RESPONSE_BASE + (next(self._tag_counter) % _TAG_RANGE)

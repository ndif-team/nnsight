"""The wire between pipeline stages.

One :class:`Link` per rank carries three kinds of traffic to and from every
other stage: values a stage's block was served (pushed by the owner as its
forward serves them, taken by the peers' blocks in the order they ask), a
block's failure (so a peer waiting on that block's values stops waiting), and
parameter or state reads answered from the owner's modules by a thread that
never touches the forward.

Per peer there is one sender thread draining an outbound queue and one receive
thread filing what arrives. Greenlets are switched only from the forward
thread, so the receive thread files items in an inbox and the forward thread
takes them; blocking on the inbox from the forward thread is safe because
nothing a peer sends waits on this rank's forward.

A message is a header of two int64 (metadata bytes, tensor bytes) and one
uint8 blob: a pickle of the envelope with every tensor replaced by a
:class:`_Slot`, then the tensors' raw bytes at aligned offsets. Two messages
because a gloo receive needs its size first; the tensors are viewed in place
over the blob on arrival. A sender that finds several envelopes queued sends
them as one message, so a step's values cost one round trip on a slow link.
"""

from __future__ import annotations

import os
import pickle
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map

# What a stage waits at most for a value a peer is bound to produce; expiry
# turns a hang into an error naming the value and its owner.
TIMEOUT_S = float(os.environ.get("NNSIGHT_PP_TIMEOUT", 60.0))

_ALIGN = 64
_TAG = 0

VALUE = "value"  # a block's read, served on the owner
CACHE = "cache"  # a cache's observation, recorded on the owner
ERROR = "error"  # the owner's block failed, or never served this value
ROUND = "round"  # the owner has run this many rounds of the request; earlier values are all sent
REQUEST = "request"  # a parameter or state read, answered from the module
REPLY = "reply"
STOP = "stop"
BATCH = "batch"  # several envelopes in one message


class _Slot:
    """A tensor's dtype, shape and byte range in the blob, standing in for it
    inside the pickled envelope."""

    def __init__(self, dtype: torch.dtype, shape: tuple, offset: int, nbytes: int) -> None:
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        self.nbytes = nbytes


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


def encode(envelope: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """``(header, blob)`` for ``envelope``; every tensor in it must be on the host."""
    slots: list = []

    def to_slot(leaf: Any) -> Any:
        if not isinstance(leaf, torch.Tensor):
            return leaf
        tensor = leaf.detach().contiguous()
        offset = _aligned(slots[-1][0].offset + slots[-1][0].nbytes) if slots else 0
        slot = _Slot(tensor.dtype, tuple(tensor.shape), offset, tensor.numel() * tensor.element_size())
        slots.append((slot, tensor))
        return slot

    meta = pickle.dumps(tree_map(to_slot, envelope))
    data_start = _aligned(len(meta))
    data_nbytes = slots[-1][0].offset + slots[-1][0].nbytes if slots else 0
    blob = torch.empty(data_start + data_nbytes, dtype=torch.uint8)
    blob[: len(meta)] = torch.frombuffer(bytearray(meta), dtype=torch.uint8)
    for slot, tensor in slots:
        if slot.nbytes:
            start = data_start + slot.offset
            blob[start : start + slot.nbytes] = tensor.view(-1).view(torch.uint8)
    return torch.tensor([len(meta), data_nbytes], dtype=torch.int64), blob


def decode(blob: torch.Tensor, meta_nbytes: int) -> dict:
    """The envelope back, its tensors viewed over ``blob`` in place."""
    envelope = pickle.loads(blob[:meta_nbytes].numpy().tobytes())
    data = blob[_aligned(meta_nbytes) :]

    def from_slot(leaf: Any) -> Any:
        if not isinstance(leaf, _Slot):
            return leaf
        return data[leaf.offset : leaf.offset + leaf.nbytes].view(leaf.dtype).reshape(leaf.shape)

    return tree_map(from_slot, envelope)


def to_host(value: Any) -> Any:
    """``value`` with every tensor copied to the host, on the calling thread's stream."""
    return tree_map(lambda t: t.detach().to("cpu", copy=True) if isinstance(t, torch.Tensor) else t, value)


class RemoteError(RuntimeError):
    """A peer reported that a value will not come, and why."""


class Link:
    """This rank's end of the wire to every other stage.

    Args:
        group: The gloo process group the stages share; ranks are group ranks.
        rank: This rank in the group.
        world: The group's size.
        resolver: ``provider -> value`` answering parameter and state requests
            from this rank's own modules, or ``None`` on a rank that answers none.
        timeout: Seconds a take or request waits before raising.
    """

    def __init__(
        self,
        group: Any,
        rank: int,
        world: int,
        resolver: Optional[Callable[[str], Any]] = None,
        timeout: float = TIMEOUT_S,
    ) -> None:
        self.group = group
        self.rank = rank
        self.world = world
        self.peers = [peer for peer in range(world) if peer != rank]
        self.resolver = resolver
        self.timeout = timeout
        self._condition = threading.Condition()
        # (request id, worker ordinal) -> provider -> what arrived for it, in order.
        self._inbox: dict[tuple, dict[str, deque]] = {}
        # (request id, worker ordinal) -> provider (or None for the whole block) -> why.
        self._errors: dict[tuple, dict[Optional[str], str]] = {}
        self._replies: dict[int, tuple[bool, Any]] = {}
        # (peer, request id) -> rounds the peer has finished for it.
        self._done: dict[tuple, int] = {}
        self._next_request = 0
        # Provider -> a parameter or module state fetched from its owner, kept
        # for the engine's life (the owner's weights do not change).
        self.kept: dict[str, Any] = {}
        self._out: dict[int, queue.Queue] = {peer: queue.Queue() for peer in self.peers}
        self._threads = []
        for peer in self.peers:
            for target in (self._send_loop, self._recv_loop):
                thread = threading.Thread(target=target, args=(peer,), daemon=True, name=f"pp-{target.__name__}-{peer}")
                thread.start()
                self._threads.append(thread)

    # ------------------------------------------------------------------ out

    def publish(self, kind: str, req: str, ordinal: int, provider: str, value: Any, selected: Any = None) -> None:
        """Send ``value`` (already on the host) to every peer, filed under the worker and provider."""
        envelope = {"kind": kind, "req": req, "ordinal": ordinal, "provider": provider, "value": value, "selected": selected}
        for peer in self.peers:
            self._out[peer].put(envelope)

    def done(self, req: str, rounds: int) -> None:
        """Tell every peer this stage has finished ``rounds`` rounds of the request,
        so a peer waiting for a value of an earlier round that never came stops
        waiting. Sent after the round's last publish, on the same queue."""
        for peer in self.peers:
            self._out[peer].put({"kind": ROUND, "req": req, "rounds": rounds})

    def fail(self, req: str, ordinal: int, provider: Optional[str], message: str) -> None:
        """Tell every peer that this worker's ``provider`` (or, with ``None``, anything of it) will not come."""
        for peer in self.peers:
            self._out[peer].put({"kind": ERROR, "req": req, "ordinal": ordinal, "provider": provider, "message": message})

    def _send_loop(self, peer: int) -> None:
        out = self._out[peer]
        while True:
            envelopes = [out.get()]
            while True:
                try:
                    envelopes.append(out.get_nowait())
                except queue.Empty:
                    break
            envelope = envelopes[0] if len(envelopes) == 1 else {"kind": BATCH, "items": envelopes}
            header, blob = encode(envelope)
            dist.send(header, group=self.group, group_dst=peer, tag=_TAG)
            dist.send(blob, group=self.group, group_dst=peer, tag=_TAG)
            if any(item["kind"] == STOP for item in envelopes):
                return

    # ------------------------------------------------------------------- in

    def _recv_loop(self, peer: int) -> None:
        while True:
            header = torch.empty(2, dtype=torch.int64)
            dist.recv(header, group=self.group, group_src=peer, tag=_TAG)
            meta_nbytes, data_nbytes = (int(n) for n in header)
            blob = torch.empty(_aligned(meta_nbytes) + data_nbytes, dtype=torch.uint8)
            dist.recv(blob, group=self.group, group_src=peer, tag=_TAG)
            envelope = decode(blob, meta_nbytes)
            for item in envelope["items"] if envelope["kind"] == BATCH else (envelope,):
                if self._file(peer, item):
                    return

    def _file(self, peer: int, envelope: dict) -> bool:
        """File one arrived envelope; True when it was the stop."""
        kind = envelope["kind"]
        if kind == STOP:
            return True
        if kind == REQUEST:
            self._answer(peer, envelope)
            return False
        with self._condition:
                if kind in (VALUE, CACHE):
                    key = (envelope["req"], envelope["ordinal"])
                    provider = envelope["provider"] if kind == VALUE else CACHE
                    self._inbox.setdefault(key, {}).setdefault(provider, deque()).append(envelope)
                elif kind == ERROR:
                    self._errors.setdefault((envelope["req"], envelope["ordinal"]), {})[envelope["provider"]] = envelope["message"]
                elif kind == ROUND:
                    self._done[(peer, envelope["req"])] = envelope["rounds"]
                elif kind == REPLY:
                    self._replies[envelope["id"]] = (envelope["ok"], envelope["value"])
                self._condition.notify_all()
        return False

    def _answer(self, peer: int, envelope: dict) -> None:
        """Answer a parameter or state request from this rank's modules.

        Runs on the receive thread. The host copy goes on a stream of its own:
        on the forward's stream it would queue behind the forward's own sends,
        which wait on the peer whose forward is waiting on this reply.
        """
        reply = {"kind": REPLY, "id": envelope["id"]}
        try:
            if self.resolver is None:
                raise AttributeError(f"stage {self.rank} answers no parameter requests")
            value = self.resolver(envelope["provider"])
            if torch.cuda.is_available():
                with torch.cuda.stream(torch.cuda.Stream()):
                    value = to_host(value)
                    torch.cuda.current_stream().synchronize()
            else:
                value = to_host(value)
            reply.update(ok=True, value=value)
        except Exception as exception:
            reply.update(ok=False, value=f"{type(exception).__name__}: {exception}")
        self._out[peer].put(reply)

    # ------------------------------------------------------------ the forward

    def has(self, req: str, ordinal: int, provider: str) -> bool:
        """Whether an item for the worker's ``provider`` has arrived (or an error for it)."""
        with self._condition:
            return self._ready(req, ordinal, provider) is not None

    def _ready(self, req: str, ordinal: int, provider: str) -> Optional[Any]:
        key = (req, ordinal)
        errors = self._errors.get(key)
        if errors and (provider in errors or None in errors):
            return errors.get(provider, errors.get(None))
        items = self._inbox.get(key, {}).get(provider)
        return items[0] if items else None

    def take(
        self,
        req: str,
        ordinal: int,
        provider: str,
        owner: Optional[int] = None,
        step: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """The next item the worker was pushed for ``provider``, waiting for it.

        Raises :class:`RemoteError` when the owner reported the value will not
        come; when ``owner`` and ``step`` are given and the owner has finished
        that step of the request without sending it (its forward ran past the
        location before the block asked); and after ``timeout`` seconds.
        """
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        with self._condition:
            while True:
                ready = self._ready(req, ordinal, provider)
                if isinstance(ready, str):
                    raise RemoteError(ready)
                if ready is not None:
                    return self._inbox[(req, ordinal)][provider].popleft()["value"]
                if owner is not None and step is not None and self._done.get((owner, req), 0) > step:
                    raise RemoteError(
                        f"'{provider}.i{step}' was requested but the model already ran past it"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RemoteError(
                        f"stage {self.rank} waited {self.timeout:.0f}s for {provider!r} of "
                        f"request {req!r} and nothing came from its owner"
                    )
                self._condition.wait(remaining)

    def take_cache(self, req: str, ordinal: int) -> list[dict]:
        """Every cache observation that has arrived for the worker, in order;
        each carries ``provider``, ``selected`` and ``value``."""
        with self._condition:
            items = self._inbox.get((req, ordinal), {}).get(CACHE)
            if not items:
                return []
            taken = list(items)
            items.clear()
            return taken

    def request(self, peer: int, provider: str, timeout: Optional[float] = None) -> Any:
        """Ask ``peer`` for ``provider`` (a parameter or a module's state) and wait for it."""
        with self._condition:
            request_id = self._next_request
            self._next_request += 1
        self._out[peer].put({"kind": REQUEST, "id": request_id, "provider": provider})
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        with self._condition:
            while request_id not in self._replies:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RemoteError(f"stage {self.rank} waited {self.timeout:.0f}s for {provider!r} from stage {peer}")
                self._condition.wait(remaining)
            ok, value = self._replies.pop(request_id)
        if not ok:
            raise RemoteError(f"stage {peer} could not answer {provider!r}: {value}")
        return value

    def drop(self, req: str) -> None:
        """Forget everything filed or kept for a finished request."""
        with self._condition:
            for key in [key for key in self._inbox if key[0] == req]:
                del self._inbox[key]
            for key in [key for key in self._errors if key[0] == req]:
                del self._errors[key]
            for key in [key for key in self._done if key[1] == req]:
                del self._done[key]

    def close(self) -> None:
        """Stop the threads: each peer's receive loop is told to return, and the
        senders return after delivering that. For tests; an engine's worker
        process exits with its daemon threads."""
        for peer in self.peers:
            self._out[peer].put({"kind": STOP})
        for thread in self._threads:
            thread.join(timeout=self.timeout)

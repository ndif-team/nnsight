"""The wire between pipeline stages.

A value a block reads on one stage reaches the others on the transfers vLLM
already makes every step. Forward, it rides the tensor dictionary each stage
hands the next (`pack` puts a step's entries into that dictionary, `unpack`
takes them out before vLLM sees it). Backward, from the last stage to the
earlier ones, it rides the broadcast the last stage makes right after sampling,
where vLLM returns the sampled tokens the same way. Either way an entry exists
for one step and arrives with that step, so nothing is waited for: a value that
is not in the step's payload was not served, and the block is told so at once.

Parameter and state reads are the one traffic a step cannot carry: a block
asks for them in the middle of a forward and needs the answer to go on. `Link`
answers those from the owner's modules by a thread that never touches the
forward, request and reply over a gloo group, one message each way.
"""

from __future__ import annotations

import os
import pickle
import queue
import threading
import time
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map

# What a stage waits at most for a parameter or state reply; expiry turns a
# hang into an error naming the value and its owner.
TIMEOUT_S = float(os.environ.get("NNSIGHT_PP_TIMEOUT", 60.0))

_ALIGN = 64
_TAG = 0

VALUE = "value"  # a block's read, served on the owner
ERROR = "error"  # the owner's block failed, or never served this value
REQUEST = "request"  # a parameter or state read, answered from the module
REPLY = "reply"
STOP = "stop"

META = "nnsight.meta"  # the key a step's entries ride under in a tensor dictionary
_TENSOR = "nnsight.t"  # prefix of the keys their tensors ride under


# ---------------------------------------------------------------- a step's entries


class _Ref:
    """Stands in for a tensor inside an entry while it rides as its own key."""

    def __init__(self, key: str) -> None:
        self.key = key


def pack(entries: list[tuple]) -> dict[str, Any]:
    """A tensor dictionary carrying ``entries``, for a transfer that takes one.

    Each entry is ``(kind, stage, req, ordinal, provider, payload)``: the
    value a stage served (``VALUE``, payload the value) or why one will not
    come (``ERROR``, payload the message). Every tensor in a payload becomes
    a key of its own, since the transfers send the dictionary's tensors
    directly and pickle everything else.
    """
    tensors: dict[str, torch.Tensor] = {}

    def to_ref(leaf: Any) -> Any:
        if not isinstance(leaf, torch.Tensor):
            return leaf
        key = f"{_TENSOR}{len(tensors)}"
        tensors[key] = leaf.detach().contiguous()
        return _Ref(key)

    meta = [(kind, stage, req, ordinal, provider, tree_map(to_ref, payload)) for kind, stage, req, ordinal, provider, payload in entries]
    return {META: meta, **tensors}


def unpack(tensor_dict: dict[str, Any]) -> list[tuple]:
    """The entries `pack` put in ``tensor_dict``, removed from it.

    Removed, so the dictionary is what its transfer's owner expects again:
    vLLM copies every key of an arrived dictionary into a buffer of its own.
    """
    meta = tensor_dict.pop(META, None)
    if meta is None:
        return []

    def from_ref(leaf: Any) -> Any:
        return tensor_dict.pop(leaf.key) if isinstance(leaf, _Ref) else leaf

    return [(kind, stage, req, ordinal, provider, tree_map(from_ref, payload)) for kind, stage, req, ordinal, provider, payload in meta]


def to_host(value: Any) -> Any:
    """``value`` with every tensor copied to the host, on the calling thread's stream."""
    return tree_map(lambda t: t.detach().to("cpu", copy=True) if isinstance(t, torch.Tensor) else t, value)


# ------------------------------------------------------ parameter and state replies


class _Slot:
    """A tensor's dtype, shape and byte range in the blob, standing in for it
    inside the pickled envelope of a reply."""

    def __init__(self, dtype: torch.dtype, shape: tuple, offset: int, nbytes: int) -> None:
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        self.nbytes = nbytes


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


def encode(envelope: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """``(header, blob)`` for ``envelope``; every tensor in it must be on the host.

    A gloo receive needs its size first, so the header carries the two
    lengths; the blob is the pickled envelope with every tensor replaced by a
    `_Slot`, then the tensors' bytes at aligned offsets.
    """
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
    """The envelope back, each tensor copied out of ``blob`` into storage of its own,
    so a value that is kept does not keep the whole message with it."""
    envelope = pickle.loads(blob[:meta_nbytes].numpy().tobytes())
    data = blob[_aligned(meta_nbytes) :]

    def from_slot(leaf: Any) -> Any:
        if not isinstance(leaf, _Slot):
            return leaf
        return data[leaf.offset : leaf.offset + leaf.nbytes].clone().view(leaf.dtype).reshape(leaf.shape)

    return tree_map(from_slot, envelope)


class RemoteError(RuntimeError):
    """A peer reported that a value will not come, and why."""


class Kept:
    """Values fetched from other stages and held on this one.

    A value fetched while a request's block runs is held while any request
    that used it runs and dropped when the last of them finishes
    (`release`); a value fetched outside any request, the head at load, is
    held for the engine's life.
    """

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.users: dict[str, set] = {}
        self.pinned: set = set()

    def __contains__(self, provider: str) -> bool:
        return provider in self.values

    def get(self, provider: str, req: Optional[str]) -> Any:
        """The kept value, now also used by ``req``."""
        if req is not None and provider not in self.pinned:
            self.users.setdefault(provider, set()).add(req)
        return self.values[provider]

    def put(self, provider: str, value: Any, req: Optional[str]) -> None:
        self.values[provider] = value
        if req is None:
            self.pinned.add(provider)
            self.users.pop(provider, None)
        else:
            self.users.setdefault(provider, set()).add(req)

    def release(self, req: str) -> None:
        """``req`` is over: drop what only it was using."""
        for provider in [provider for provider, reqs in self.users.items() if req in reqs]:
            self.users[provider].discard(req)
            if not self.users[provider]:
                del self.users[provider]
                del self.values[provider]


class Link:
    """This rank's end of the request-and-reply wire to every other stage.

    Args:
        group: The gloo process group the stages share; ranks are group ranks.
        rank: This rank in the group.
        world: The group's size.
        resolver: ``provider -> value`` answering parameter and state requests
            from this rank's own modules, or ``None`` on a rank that answers none.
        timeout: Seconds a request waits before raising.
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
        # Ids someone is still waiting for; a reply to any other id is dropped
        # as it arrives, so a waiter that gave up leaves nothing behind.
        self._waiting: set[int] = set()
        self._replies: dict[int, tuple[bool, Any]] = {}
        self._next_request = 0
        # Parameters and module states fetched from their owners (see Kept).
        self.kept = Kept()
        self._out: dict[int, queue.Queue] = {peer: queue.Queue() for peer in self.peers}
        self._threads = []
        for peer in self.peers:
            for target in (self._send_loop, self._recv_loop):
                thread = threading.Thread(target=target, args=(peer,), daemon=True, name=f"pp-{target.__name__}-{peer}")
                thread.start()
                self._threads.append(thread)

    # Each loop handles one message per call of a method, so what a message
    # referenced is released when the method returns, before the loop blocks
    # for the next one; a thread waiting for a message holds none.

    def _send_loop(self, peer: int) -> None:
        while not self._send_next(peer):
            pass

    def _send_next(self, peer: int) -> bool:
        envelope = self._out[peer].get()
        header, blob = encode(envelope)
        dist.send(header, group=self.group, group_dst=peer, tag=_TAG)
        dist.send(blob, group=self.group, group_dst=peer, tag=_TAG)
        return envelope["kind"] == STOP

    def _recv_loop(self, peer: int) -> None:
        while not self._recv_next(peer):
            pass

    def _recv_next(self, peer: int) -> bool:
        header = torch.empty(2, dtype=torch.int64)
        dist.recv(header, group=self.group, group_src=peer, tag=_TAG)
        meta_nbytes, data_nbytes = (int(n) for n in header)
        blob = torch.empty(_aligned(meta_nbytes) + data_nbytes, dtype=torch.uint8)
        dist.recv(blob, group=self.group, group_src=peer, tag=_TAG)
        envelope = decode(blob, meta_nbytes)
        kind = envelope["kind"]
        if kind == STOP:
            return True
        if kind == REQUEST:
            self._answer(peer, envelope)
        elif kind == REPLY:
            self._file_reply(envelope["id"], envelope["ok"], envelope["value"])
        return False

    def _file_reply(self, request_id: int, ok: bool, value: Any) -> None:
        """Hand a reply to its waiter, or drop it when nobody is waiting any more."""
        with self._condition:
            if request_id in self._waiting:
                self._replies[request_id] = (ok, value)
                self._condition.notify_all()

    def _answer(self, peer: int, envelope: dict) -> None:
        """Answer a parameter or state request from this rank's modules.

        Runs on the receive thread. The host copy goes on a stream of its own:
        on the forward's stream it would queue behind the forward's own work,
        which may be waiting on the peer whose forward is waiting on this reply.
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

    def request(self, peer: int, provider: str, timeout: Optional[float] = None) -> Any:
        """Ask ``peer`` for ``provider`` (a parameter or a module's state) and wait for it."""
        with self._condition:
            request_id = self._next_request
            self._next_request += 1
            self._waiting.add(request_id)
        try:
            self._out[peer].put({"kind": REQUEST, "id": request_id, "provider": provider})
            deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
            with self._condition:
                while request_id not in self._replies:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RemoteError(f"stage {self.rank} waited {self.timeout:.0f}s for {provider!r} from stage {peer}")
                    self._condition.wait(remaining)
                ok, value = self._replies.pop(request_id)
        finally:
            with self._condition:
                self._waiting.discard(request_id)
                self._replies.pop(request_id, None)
        if not ok:
            raise RemoteError(f"stage {peer} could not answer {provider!r}: {value}")
        return value

    def drop(self, req: str) -> None:
        """Forget what was kept for a finished request."""
        self.kept.release(req)

    def close(self) -> None:
        """Stop the threads: each peer's receive loop is told to return, and the
        senders return after delivering that. For tests; an engine's worker
        process exits with its daemon threads."""
        for peer in self.peers:
            self._out[peer].put({"kind": STOP})
        for thread in self._threads:
            thread.join(timeout=self.timeout)

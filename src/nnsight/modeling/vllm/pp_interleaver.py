"""The interleaver of one pipeline stage.

Every stage runs the whole block. This interleaver serves the modules the
stage holds as any interleaver does, and for the rest: what its own workers
are served here goes into the step's outbox (`serving`) and rides vLLM's own
transfers to the other stages (see `pp_transport`), and what its workers ask
of other stages is taken from the inbox those transfers fill, in the order
they ask (`chase`, `serve`). A worker asking for an earlier stage's value is
answered in place, since that stage's step, and its payload, came before this
one; a worker asking for a later stage's value parks until that stage's
values for the step arrive, which is before this stage's next step. Nothing
is waited for beyond that: a value missing from a step's payload was not
served in that step, and the worker is told so.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional

import torch
from greenlet import getcurrent
from torch.utils._pytree import tree_map

from ...intervention.interleaver import Event, Mediator
from .interleaver import VLLMInterleaver
from .pp_transport import ERROR, VALUE, Link, RemoteError

ROUND = "round"  # the last stage has run this step for the request; its values for it are in this payload


def _clone(value: Any) -> Any:
    return tree_map(lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t, value)


class PPInterleaver(VLLMInterleaver):
    """A [`VLLMInterleaver`][nnsight.modeling.vllm.interleaver.VLLMInterleaver] for one stage of a pipeline.

    Attributes:
        module_map: Which stage holds each module path.
        link: The request-and-reply wire to the other stages, for parameters and state.
        local_rank: This stage.
        device: Where a taken value is placed before the worker gets it.
        rounds: Request id -> how many forwards this stage has run for it.
        arrived: Request id -> how many of its steps the later stages' values have arrived for.
    """

    def __init__(
        self,
        module_map: Any,
        link: Link,
        local_rank: int,
        device: torch.device,
        taps: Iterable[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(taps, **kwargs)
        self.module_map = module_map
        self.link = link
        self.local_rank = local_rank
        self.device = device
        self.rounds: dict[str, int] = {}
        self.arrived: dict[str, int] = {}
        # This step's entries for the other stages, and the ones that arrived
        # from earlier stages this step, which the stages after this one need too.
        self.outbox: list[tuple] = []
        self.relayed: list[tuple] = []
        # (request id, worker ordinal) -> provider -> what arrived for it, in order.
        self.inbox: dict[tuple, dict[str, deque]] = {}
        # (request id, worker ordinal) -> provider (or None for the whole block) -> why it will not come.
        self.errors: dict[tuple, dict[Optional[str], str]] = {}

    # ---------------------------------------------------------------- where

    def owner(self, provider: str) -> Optional[int]:
        """The stage holding ``provider``'s module, or ``None`` when it is this one."""
        owner = self.module_map.get_owning_rank(provider)
        return None if owner is None or owner == self.local_rank else owner

    @staticmethod
    def _key(mediator: Mediator) -> tuple[str, int]:
        return mediator.pp_req, mediator.pp_ordinal

    # ------------------------------------------------------------ the owner

    def serving(
        self,
        mediator: Mediator,
        provider: str,
        value: Any,
        line: Optional[int] = None,
    ) -> None:
        """Put what this stage is about to serve its worker in the step's outbox.

        Copied now, before the worker gets it: the block may change the value
        in place before it parks again (vLLM's fused norm writes into its
        inputs), and the peers must get what the worker was handed.
        """
        if self.owner(provider) is None and self.link.peers:
            self.outbox.append((VALUE, self.local_rank, *self._key(mediator), provider, _clone(value)))

    def failed(self, mediator: Mediator, message: str) -> None:
        """This worker will serve nothing more: tell the peers' copies waiting on it."""
        self.outbox.append((ERROR, self.local_rank, *self._key(mediator), None, message))

    def served(
        self,
        mediator: Mediator,
        provider: str,
        value: Any,
        selected: Optional[tuple] = None,
        event: Event = Event.VALUE,
        line: Optional[int] = None,
    ) -> None:
        """The worker has parked again: note the step, then answer what it asks next.

        A cache observation (``selected``) stays where it was made: each
        stage's cache keeps what its own modules produced, and the caches are
        unioned at collect, so this stage's saves go home for that.
        """
        if selected is not None:
            if self.owner(provider) is None:
                mediator.pp_reports = True
            return
        if self.owner(provider) is None:
            # A local visit is served in the round being run, so that is the
            # step the block is at, whatever module fired.
            mediator.pp_step = self.rounds.get(mediator.pp_req, 0)
        self.chase(mediator)

    def started(self, mediator: Mediator) -> None:
        """A worker has started and parked for the first time."""
        self.chase(mediator)

    # ------------------------------------------------------------- the wire

    def flush_forward(self) -> list[tuple]:
        """The entries for the next stage: this step's, and the earlier stages'
        that arrived this step, which the stages after this one have not seen."""
        entries = self.outbox + self.relayed
        self.outbox, self.relayed = [], []
        return entries

    def flush_backward(self, reqs: Iterable[str]) -> list[tuple]:
        """The last stage's entries for the earlier ones after a step: what it
        and the stages between served (see `receive` for which those are),
        and a mark per request that the step is done, so a worker waiting for
        a value of it stops waiting."""
        entries = self.outbox + self.relayed + [(ROUND, self.local_rank, req, 0, None, None) for req in reqs]
        self.outbox, self.relayed = [], []
        return entries

    def receive(self, entries: list[tuple], forward: bool) -> None:
        """File a payload's entries. Forward, the ones from earlier stages are
        this stage's and are kept for the stages after it; backward, the ones
        from later stages. Each value is filed once whichever way it came.

        What is kept to send on is a copy taken now: the filed value is handed
        to this stage's copy of the block, which may change it in place before
        the send. A stage before the last sends on everything it got forward;
        the last sends back only the stages between's, since the first stage's
        reached every later stage forward and no stage before it needs them.
        """
        last = self.local_rank == self.link.world - 1
        for entry in entries:
            kind, stage, req, ordinal, provider, payload = entry
            if (stage < self.local_rank) != forward or stage == self.local_rank:
                continue
            key = (req, ordinal)
            if kind == VALUE:
                self.inbox.setdefault(key, {}).setdefault(provider, deque()).append(payload)
            elif kind == ERROR:
                self.errors.setdefault(key, {})[provider] = payload
            elif kind == ROUND:
                self.arrived[req] = self.arrived.get(req, 0) + 1
            if forward and (not last or stage > 0):
                self.relayed.append((kind, stage, req, ordinal, provider, _clone(payload)))

    # --------------------------------------------------------- the receiver

    def _step(self, mediator: Mediator) -> int:
        """The step of the run the worker's block is at.

        Two sources: a local visit is served in the round being run, which
        `served` records; and ``tracer.iter`` pins a worker's requests to a
        step, read off the pending request while the pin stands (it is relaxed
        after its first hit, and the requests that follow in the same step
        carry occurrence counts instead). A block with no ``tracer.iter`` and
        no local visit yet is at step 0.
        """
        pending = mediator.pending
        if mediator.iteration is not None and pending is not None and pending.iteration is not None:
            mediator.pp_step = pending.iteration
        return mediator.pp_step

    def _produced(self, mediator: Mediator, owner: int) -> bool:
        """Whether ``owner``'s values for the step the worker is at are here.

        An earlier stage's values for a step arrive with the step, before this
        stage's forward; a later stage's arrive after that stage's forward of
        the step, before this stage's next one, and `arrived` counts them.
        """
        step = self._step(mediator)
        if owner > self.local_rank:
            return step < self.arrived.get(mediator.pp_req, 0)
        return step <= self.rounds.get(mediator.pp_req, 0)

    def chase(self, mediator: Mediator) -> None:
        """Answer the worker's requests for other stages' modules as far as possible now.

        A write to another stage's module is absorbed (the owner applies the
        same line). A read is answered from the inbox once the owner's values
        for that step are here (see `_produced`); otherwise the worker stays
        parked for `serve` to answer at a later step.
        """
        while mediator.alive and (pending := mediator.pending) is not None and pending.provider is not None:
            owner = self.owner(pending.provider)
            if owner is None:
                return
            if pending.event in (Event.SWAP, Event.SKIP):
                mediator.pending = mediator.switch()
                continue
            if pending.event is not Event.VALUE or not self._produced(mediator, owner):
                return
            self._take(mediator, owner)

    def _take(self, mediator: Mediator, owner: int) -> None:
        """Hand the worker the next item filed for its read, or why there is none."""
        provider = mediator.pending.provider
        key = self._key(mediator)
        items = self.inbox.get(key, {}).get(provider)
        if items:
            value = items.popleft()
            self._hand(mediator, tree_map(lambda t: t.to(self.device) if isinstance(t, torch.Tensor) else t, value))
            return
        errors = self.errors.get(key, {})
        message = errors.get(provider, errors.get(None))
        if message is None:
            message = (
                f"'{provider}' was requested at step {self._step(mediator)} of request "
                f"{mediator.pp_req!r}, but stage {owner} served nothing for it in that step: "
                "its copy of the block did not reach that read, or the model ran past it"
            )
        self._hand(mediator, RemoteError(message))

    def _hand(self, mediator: Mediator, value: Any) -> None:
        """Resume the worker with ``value`` (thrown into it when it is a RemoteError)."""
        mediator.worker.parent = getcurrent()
        # A pinned step relaxes on its first hit, as it does when a local visit
        # serves it, so the step's later requests follow the model.
        if mediator.iteration:
            mediator.iteration = None
        try:
            if isinstance(value, RemoteError):
                mediator.pending = mediator.worker.throw(value)
            else:
                mediator.pending = mediator.switch(value)
        except Exception as exception:
            # The block raised on the way: deferred to its own request, as a
            # raise out of a module's handoff is (see Interleaver.handle).
            if not self.defer_exceptions:
                raise
            mediator.exception = exception
            mediator.pending = None

    def serve(self, mediators: Iterable[Mediator]) -> None:
        """Resume workers parked on other stages' values that have arrived.

        Called at the start of a step for the workers of the requests it runs.
        """
        for mediator in mediators:
            if not mediator.alive or (pending := mediator.pending) is None or pending.provider is None:
                continue
            owner = self.owner(pending.provider)
            if owner is None or pending.event is not Event.VALUE or not self._produced(mediator, owner):
                continue
            self._take(mediator, owner)
            self.chase(mediator)

    def finished(self, req: str) -> None:
        """Forget a request: its rounds here, what was filed for it, and what was kept for it."""
        self.rounds.pop(req, None)
        self.arrived.pop(req, None)
        for key in [key for key in self.inbox if key[0] == req]:
            del self.inbox[key]
        for key in [key for key in self.errors if key[0] == req]:
            del self.errors[key]
        self.link.drop(req)

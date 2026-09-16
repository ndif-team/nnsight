"""The interleaver of one pipeline stage.

Every stage runs the whole block. This interleaver serves the modules the
stage holds as any interleaver does, and for the rest: what its own workers
are served here is pushed to the other stages as it happens (`served`), and
what its workers ask of other stages is taken from the link's inbox, in the
order they ask (`chase`, `serve`). A worker asking for a value an earlier
stage holds is answered in place, since that stage has already produced it; a
worker asking for a later stage's value parks until this stage's next step,
by which time that value exists. See ``docs/developing/pp-push-design.md``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from greenlet import getcurrent

from ...intervention.interleaver import Event, Mediator
from .interleaver import VLLMInterleaver
from .pp_transport import CACHE, VALUE, Link, RemoteError, to_host


class PPInterleaver(VLLMInterleaver):
    """A [`VLLMInterleaver`][nnsight.modeling.vllm.interleaver.VLLMInterleaver] for one stage of a pipeline.

    Attributes:
        module_map: Which stage holds each module path.
        link: The wire to the other stages.
        local_rank: This stage.
        device: Where a taken value is placed before the worker gets it.
        rounds: Request id -> how many forwards this stage has run for it.
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

    # ---------------------------------------------------------------- where

    def owner(self, provider: str) -> Optional[int]:
        """The stage holding ``provider``'s module, or ``None`` when it is this one."""
        owner = self.module_map.get_owning_rank(provider)
        return None if owner is None or owner == self.local_rank else owner

    @staticmethod
    def _key(mediator: Mediator) -> tuple[str, int]:
        return mediator.pp_req, mediator.pp_ordinal

    # ------------------------------------------------------------ the owner

    def served(
        self,
        mediator: Mediator,
        provider: str,
        value: Any,
        selected: Optional[tuple] = None,
        event: Event = Event.VALUE,
    ) -> None:
        """Push what this stage just served, then answer what the worker asks next."""
        key = self._key(mediator)
        if self.owner(provider) is None:
            # A local visit is served in the round being run, so that is the
            # step the block is at, whatever module fired.
            mediator.pp_step = self.rounds.get(key[0], 0)
            if event is Event.VALUE and self.link.peers:
                self.link.publish(CACHE if selected else VALUE, *key, provider, to_host(value), selected)
        if selected is None:
            self._report_stale(mediator)
            self.chase(mediator)

    def _report_stale(self, mediator: Mediator) -> None:
        """A worker asking for a local visit the forward already made will never
        be served; tell the peers waiting on it, so they fail where it failed
        rather than at their deadline. The worker itself stays parked and is
        reported when its request ends, as it is on one GPU."""
        pending = mediator.pending
        if pending is None or pending.event is not Event.VALUE or pending.provider is None:
            return
        if self.owner(pending.provider) is not None or pending.iteration is None:
            return
        if pending.iteration < mediator.occurrence(pending.provider):
            self.link.fail(
                *self._key(mediator),
                pending.provider,
                f"'{pending}' was requested but the model already ran past it",
            )

    def started(self, mediator: Mediator) -> None:
        """A worker has started and parked for the first time."""
        self.chase(mediator)

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

    def _produced(self, mediator: Mediator, owner: int, final: bool) -> bool:
        """Whether ``owner`` has produced what the worker is asking for.

        The stages run a request's rounds in order: while this stage runs
        round ``n`` (``rounds`` forwards done here), an earlier stage has
        finished round ``n`` and a later one round ``n-1``. Once the request is
        over (``final``) every round that ran is produced everywhere, and a
        step past the last round never will be.
        """
        rounds = self.rounds.get(mediator.pp_req, 0)
        step = self._step(mediator)
        if final or owner > self.local_rank:
            return step < rounds
        return step <= rounds

    def chase(self, mediator: Mediator, final: bool = False) -> None:
        """Answer the worker's requests for other stages' modules as far as possible now.

        A write to another stage's module is absorbed (the owner applies the
        same line). A read is taken from the inbox in place when the owner has
        produced it (see `_produced`), which for an earlier stage's value in
        the round being run is always; otherwise the worker stays parked for
        `serve` to answer at a later step.
        """
        while mediator.alive and (pending := mediator.pending) is not None and pending.provider is not None:
            owner = self.owner(pending.provider)
            if owner is None:
                return
            if pending.event in (Event.SWAP, Event.SKIP):
                mediator.pending = mediator.switch()
                continue
            if pending.event is not Event.VALUE or not self._produced(mediator, owner, final):
                return
            self._take(mediator, pending.provider)
            self._report_stale(mediator)

    def _take(self, mediator: Mediator, provider: str) -> None:
        """Hand the worker the next pushed item for ``provider``, or the owner's failure."""
        key = self._key(mediator)
        owner, step = self.owner(provider), self._step(mediator)
        mediator.worker.parent = getcurrent()
        # A pinned step relaxes on its first hit, as it does when a local visit
        # serves it, so the step's later requests follow the model.
        if mediator.iteration:
            mediator.iteration = None
        try:
            try:
                value = self.link.take(*key, provider, owner, step)
            except RemoteError as error:
                mediator.pending = mediator.worker.throw(error)
                return
            value = torch.utils._pytree.tree_map(
                lambda t: t.to(self.device) if isinstance(t, torch.Tensor) else t, value
            )
            mediator.pending = mediator.switch(value)
        except Exception as exception:
            # The block raised on the way: deferred to its own request, as a
            # raise out of a module's handoff is (see Interleaver.handle).
            if not self.defer_exceptions:
                raise
            mediator.exception = exception
            mediator.pending = None

    def serve(self, mediators: Iterable[Mediator], final: bool = False) -> None:
        """Resume workers parked on other stages' values that are now produced.

        Called at the start of a step for the workers of the requests it runs,
        and at collect (``final``) for a finished request's workers. Cache
        observations that arrived for the workers are recorded on their caches.
        """
        for mediator in mediators:
            key = self._key(mediator)
            for item in self.link.take_cache(*key):
                for cache in mediator.caches:
                    if item["provider"] in cache.subscriptions():
                        cache.observe_selected(item["selected"], item["value"])
            if not mediator.alive or (pending := mediator.pending) is None or pending.provider is None:
                continue
            owner = self.owner(pending.provider)
            if owner is None or pending.event is not Event.VALUE or not self._produced(mediator, owner, final):
                continue
            self._take(mediator, pending.provider)
            self._report_stale(mediator)
            self.chase(mediator, final)

    def finished(self, req: str) -> None:
        """Forget a request: its rounds here and everything the link filed for it."""
        self.rounds.pop(req, None)
        self.link.drop(req)

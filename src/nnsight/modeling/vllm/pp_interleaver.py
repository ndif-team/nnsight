"""The pipeline-parallel interleaver: intercept, publish, serve.

Under PP every rank runs every intervention block, but each rank's forward only
visits its own stage's modules. This interleaver closes the gap in three moves:

* **Intercept** (worker side, via the core
  :meth:`~nnsight.intervention.interleaver.Interleaver.intercept` seam): a read
  of a remote-owned location is answered immediately with a
  :class:`~.lazy_remote_tensor.LazyRemoteTensor` (no traffic, worker keeps
  running); a write to one is absorbed (the owning rank runs the same line
  locally). A worker forcing an *upstream*-owned lazy is served in place: the
  value already exists (pipeline order), so the intercept blocks on the
  transfer and returns it without parking, preserving the worker's ability to
  write locations the forward has not reached yet. Forcing a *downstream*-owned
  lazy parks on its encoded pull location — the intercept issues the
  cross-stage pull at that exact moment (issue-at-park: the transfer overlaps
  the rest of the forward) and lets the park stand.

* **Publish** (producer side): as :meth:`serve` offers a location the rank
  owns, each request's rows of the post-intervention value are cloned into the
  pull buffer under ``("{provider}.i{visit}", req_id)`` and any parked peer
  pulls for that key are dispatched. ``serve`` runs inside ``handle``'s
  fragments bracket, so under TP within a stage the published value is the
  assembled whole, matching what a local worker read.

* **Serve** (driver side): at a serve point — after the local forward, before
  the next step — :meth:`serve_pulls` completes each worker's in-flight pull
  and switches the worker back in with the value. A resumed worker may force
  another lazy and re-park; the loop drains until no worker waits on a pull.

Occurrence tags: the buffer key's ``.i{visit}`` must agree between the owning
rank (which tags by its mediator's visit count for the location) and a
non-owning rank (whose forward never visits it). The non-owning side tags a
pinned read (``tracer.iter``) with the pinned step, and a relaxed read with
:attr:`step` — the interleaver's forward count, advanced by the runner once per
``execute_model`` — which equals the visit count for modules the forward
reaches once per step. A module visited several times in one forward is
mis-tagged on the non-owning side (same limitation as the 0.7 branch).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils._pytree import tree_map

from ...intervention.interleaver import STEP_GATE, Event, Interleaver, Mediator
from .lazy_remote_tensor import (
    PULL_LOCATION_PREFIX,
    LazyRemoteTensor,
    decode_pull_location,
    encode_pull_location,
)
from .pp import PPModuleMap, resolve_meta


def _occurrence(provider: str) -> Optional[int]:
    """The ``.i{n}`` occurrence a tagged provider names, or ``None``."""
    _, _, tag = provider.rpartition(".")
    if tag.startswith("i") and tag[1:].isdigit():
        return int(tag[1:])
    return None


def _strip_park_tag(location: str) -> str:
    """Drop the ``.i{n}`` tag :meth:`Mediator.event` appended to a park."""
    head, _, tail = location.rpartition(".")
    if tail.startswith("i") and tail[1:].isdigit():
        return head
    return location


class PPInterleaver(Interleaver):
    """An interleaver for one rank of a pipeline-parallel model.

    Attributes:
        module_map: Path → owning-stage resolution (derived at load).
        module_meta: Per-module metadata from the load-time exchange (dtype
            hints for lazies), keyed by raw ``named_modules()`` names.
        listener: This rank's :class:`~.pp_listener.PPListener`.
        local_rank: This rank's PP stage index.
        step: Forward count, advanced by the runner once per ``execute_model``;
            the occurrence tag for relaxed remote reads and published values of
            once-per-step modules.
    """

    def __init__(
        self,
        module_map: PPModuleMap,
        listener: Any,
        local_rank: int,
        module_meta: Optional[dict] = None,
        fragments: Optional[Any] = None,
    ) -> None:
        # Fragments (the within-stage TP gather) ride the base bracket
        # unchanged: `handle` assembles the whole before `serve` runs, so both
        # the local workers and the publish below see the real tensor.
        super().__init__(fragments=fragments)
        self.module_map = module_map
        self.module_meta = module_meta if module_meta is not None else {}
        self.listener = listener
        self.local_rank = local_rank
        self.step = 0
        # Completed forward rounds per request id, maintained by the runner.
        # The pipeline schedule makes this the local ground truth for whether
        # a pulled value exists yet: the engine schedules a request's round k
        # only after round k-1 sampled, and sampling runs on the last stage
        # after EVERY stage finished round k-1 — so when this rank opens
        # round k, all stages have completed rounds 0..k-1 for the request.
        self.rounds: dict = {}
        # In-flight pulls keyed by (id(mediator), untagged park location).
        self._pulls: dict[tuple[int, str], Any] = {}

    # ------------------------------------------------------------------
    # Worker side: the intercept
    # ------------------------------------------------------------------

    def intercept(
        self, mediator: Mediator, event: Event, location: str, rest: tuple
    ) -> tuple | None:
        # A forced lazy parking for its value.
        if location.startswith(PULL_LOCATION_PREFIX):
            source_rank, req_id, provider = decode_pull_location(location)
            # An upstream-owned value from a round this rank has opened
            # already exists (pipeline order: the earlier stage finishes a
            # round before ours starts it), so the wait is transfer only.
            # Serve it in place — blocking the worker right here, inside
            # whatever switched it in — instead of parking. Parking would
            # surrender the swap window: the worker could only resume at a
            # serve point outside the forward, after the model ran past every
            # location the rest of the block might write (a write to the very
            # module whose hook is live underneath this force raises
            # OutOfOrderError once resumed late). An error or timeout raises
            # here, on the worker, at the line that forced the value.
            # A pull for a round this rank has NOT opened (a loop body running
            # ahead of the model) parks like a downstream pull below and is
            # resumed by the serve point once that round has run.
            if source_rank < self.local_rank:
                occurrence = _occurrence(provider)
                rounds = (
                    self.rounds.get(req_id) if req_id is not None else None
                )
                if (
                    rounds is None
                    or occurrence is None
                    or occurrence <= rounds
                ):
                    pull = self.listener.begin_pull(source_rank, provider, req_id)
                    return (pull.complete(),)
            # Downstream-owned: the value is produced only after this rank's
            # forward returns, so blocking here would deadlock the pipeline.
            # Issue the pull NOW, on this worker's way into the park, so the
            # transfer overlaps the remainder of the forward (issue-at-park;
            # deferring the send to the serve point would serialize the wire
            # time onto every step). The park stands — serve_pulls resumes
            # the worker once the value has arrived.
            key = (id(mediator), location)
            if key not in self._pulls:
                self._pulls[key] = self.listener.begin_pull(
                    source_rank, provider, req_id
                )
            return None

        # BARRIER carries no location; local locations park normally.
        if location is None or self.module_map.is_local(location, self.local_rank):
            return None

        owner = self.module_map.get_owning_rank(location)
        if event is Event.VALUE:
            # Tag the read with the occurrence the owning rank will publish
            # under: the pinned step inside ``tracer.iter`` (kept for every
            # read of the step's body, so they all name the same round), else
            # the request's completed-round count. The owning rank's visit
            # counter for a once-per-step module advances one visit per round
            # of the request, so the two clocks agree. A run outside the
            # engine (no request id) counts its own forwards instead.
            if mediator.iteration is not None:
                occurrence = mediator.iteration
            else:
                req_id = self._req_id(mediator)
                rounds = (
                    self.rounds.get(req_id) if req_id is not None else None
                )
                occurrence = rounds if rounds is not None else self.step
            provider = f"{location}.i{occurrence}"
            lazy = LazyRemoteTensor(
                owner,
                provider,
                self._dtype_hint(location),
                req_id=self._req_id(mediator),
            )
            return (lazy,)
        # A write (SWAP) or skip to a remote-owned module: absorbed — the
        # owning rank executes the same block line against the real module.
        return (None,)

    def _req_id(self, mediator: Mediator) -> Optional[str]:
        """The vLLM request id this worker rides, stamped by the runner at
        request ingest; ``None`` for a single-trace (non-engine) run."""
        return getattr(mediator, "pp_req_id", None)

    def _dtype_hint(self, location: str) -> Any:
        meta = resolve_meta(
            self.module_meta, _strip_eproperty_key(location), self.module_map.root_path
        )
        if meta is None:
            return None
        return meta.get("dtype")

    # ------------------------------------------------------------------
    # Producer side: publish owned values into the pull buffer
    # ------------------------------------------------------------------

    def serve(self, provider: str, value: Any) -> Any:
        # Publish exactly what some worker is waiting on. Every rank runs the
        # same block, so a location a peer pulls is one this rank's copy of
        # the worker parks on locally; a worker's park already carries the
        # occurrence tag of the visit it wants, so this is the same string
        # match handle() serves by, taken BEFORE the mediators run (their
        # handle() advances the visit count). The step gate carries no data
        # and locations nothing reads are never buffered.
        wanted = (
            [
                (mediator, visit)
                for mediator, visit in (
                    (mediator, mediator.iterations[provider])
                    for mediator in self.mediators
                )
                if mediator.pending is not None
                and mediator.pending[1] == f"{provider}.i{visit}"
            ]
            if (
                provider != STEP_GATE
                and self.listener is not None
                and self.module_map.is_local(provider, self.local_rank)
            )
            else []
        )
        value = super().serve(provider, value)
        # Publishing from serve — inside handle's gather bracket — is what puts
        # the post-intervention value into the pull buffer as the workers saw
        # it: under TP-within-a-stage that is the assembled whole, where
        # handle's return value is already re-split for the model.
        if wanted:
            self._publish(provider, value, wanted)
        return value

    def _publish(self, provider: str, value: Any, visits: list) -> None:
        """Clone each request's rows of the post-intervention value into the
        pull buffer and dispatch any parked peer pulls for them.

        Values are published per request (the wire key a consumer pulls with is
        ``(tagged_provider, req_id)``), narrowed to that request's rows exactly
        as its own worker's reads are. A whole-batch worker (no request id)
        publishes under ``req_id=None``.
        """
        buffer = self.listener._buffer
        condition = self.listener._condition
        for mediator, visit in visits:
            tagged = f"{provider}.i{visit}"
            req_id = self._req_id(mediator)
            served = (
                value
                if self.batcher is None
                else self.batcher.narrow(value, mediator.batch_group)
            )
            # Clone: the buffer must hold the value as of this visit — the
            # model (or a later intervention) may mutate the live tensor.
            cloned = tree_map(
                lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t,
                served,
            )
            key = (tagged, req_id)
            with condition:
                buffer[key] = cloned
            self.listener.dispatch_parked(key, cloned)

    # ------------------------------------------------------------------
    # Driver side: the serve point
    # ------------------------------------------------------------------

    def serve_pulls(
        self,
        block: bool = True,
        drain: bool = True,
        only: Optional[set] = None,
    ) -> None:
        """Resume workers parked on pulls.

        Called at serve points. ``block=True`` (collect/finalize, sampling)
        completes every parked worker's pull, waiting for in-flight transfers.
        ``block=False`` (the end of a step, while the pipeline is still moving)
        serves only pulls whose value has already arrived.

        ``only`` scopes the serve to the workers riding the named requests
        (collect passes the finished request ids); every other worker keeps its
        park and is resumed by its own request's step serves.

        ``drain=False`` (the start of a step) blocks only on pulls whose
        target the pipeline has already produced: a pull's occurrence tag and
        the requester's completed-round count share the sampling-round clock,
        so ``occurrence < rounds`` means the producing round finished and the
        wait is transfer only. A pull for the current or a later round is left
        parked — its value is produced by forwards this serve point must not
        delay; blocking on it inverts the pipeline order into a deadlock (a
        worker chaining per-step forces under ``tracer.iter`` re-parks here on
        the NEXT round's value). A worker several rounds behind still catches
        up fully in one call: each chained pull it re-parks on is a past round
        until it reaches the current one.

        A resumed worker may immediately force another lazy: its intercept
        issues the new pull and re-parks, so the loop drains until no worker
        waits on a servable pull.
        """
        progressed = True
        while progressed:
            progressed = False
            for mediator in list(self.mediators):
                if only is not None and self._req_id(mediator) not in only:
                    continue
                if not mediator.alive or mediator.pending is None:
                    continue
                if mediator.pending[0] is not Event.VALUE:
                    continue
                untagged = _strip_park_tag(mediator.pending[1])
                if not untagged.startswith(PULL_LOCATION_PREFIX):
                    continue
                key = (id(mediator), untagged)
                pull = self._pulls.get(key)
                if pull is None:
                    continue
                if not block and not pull.ready:
                    continue
                if not drain and not pull.ready:
                    _, req_id, provider = decode_pull_location(untagged)
                    rounds = self.rounds.get(req_id)
                    if rounds is not None:
                        occurrence = _occurrence(provider)
                        # An unparseable occurrence is treated as current-round
                        # (left parked): its transfer is in flight and the next
                        # boundary completes it, which is always safe.
                        if occurrence is None or occurrence >= rounds:
                            continue
                del self._pulls[key]
                try:
                    value = pull.complete()
                    mediator.pending = mediator.switch(value)
                except Exception as exception:
                    # Two shapes land here. The pull failed (error reply,
                    # timeout): the worker is still parked, so throw the error
                    # into it at the line that forced the value — a worker
                    # that catches it parks again (keep its new park); one
                    # that doesn't is unwound (its finally blocks run) and the
                    # error is recorded. Or the resumed worker itself raised
                    # out of switch(): it is already unwound — just record.
                    # Either way, like handle(), tear down only when not
                    # deferring: on a shared engine this ends one request.
                    if mediator.alive:
                        try:
                            mediator.pending = mediator.worker.throw(exception)
                            progressed = True
                            continue  # the worker recovered; not an error
                        except BaseException:
                            pass
                    mediator.pending = None
                    mediator.exception = exception
                    if not self.defer_exceptions:
                        raise
                progressed = True

    def has_pending_pulls(self) -> bool:
        """Whether any worker is currently parked on a pull location."""
        for mediator in self.mediators:
            if (
                mediator.alive
                and mediator.pending is not None
                and mediator.pending[1] is not None
                and _strip_park_tag(mediator.pending[1]).startswith(
                    PULL_LOCATION_PREFIX
                )
            ):
                return True
        return False


def _strip_eproperty_key(location: str) -> str:
    """Drop a trailing ``output``/``input``/``inputs`` for metadata lookup."""
    head, _, tail = location.rpartition(".")
    if tail in ("output", "input", "inputs"):
        return head
    return location

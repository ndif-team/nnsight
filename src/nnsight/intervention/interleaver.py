"""Interleaving intervention code with a model's forward pass.

nnsight lets you read and edit a model's intermediate values from ordinary
Python written *inside* a ``with model.trace(...):`` block. To make that work,
the intervention code and the model's forward pass have to run in lockstep:
the intervention pauses whenever it asks for a value the model hasn't produced
yet, the model runs until it reaches that value, hands it over, and the
intervention resumes — possibly editing the value on the way back in.

This module implements that dance with `greenlets <https://greenlet.readthedocs.io>`_
(cooperative, single-threaded coroutines), not OS threads:

* Each block of intervention code becomes a [`Mediator`][nnsight.intervention.interleaver.Mediator], which runs the
  code in its own greenlet — the "worker". The worker drives the interaction:
  it runs until it needs a value, then *parks*, switching control back to the
  greenlet that started it (the "parent", i.e. the model side).

* The worker parks by naming a **location** — a provider string such as
  ``"model.layer1.output"`` or the run's ``"result"``. It parks to *read* a
  location ([`Mediator.value`][nnsight.intervention.interleaver.Mediator.value]), to *replace* one ([`Mediator.swap`][nnsight.intervention.interleaver.Mediator.swap]),
  or to *skip* a gated computation ([`Mediator.skip`][nnsight.intervention.interleaver.Mediator.skip]). It can also park on
  no location at all, waiting on the other workers rather than the model
  ([`Mediator.barrier`][nnsight.intervention.interleaver.Mediator.barrier]).

* An [`Interleaver`][nnsight.intervention.interleaver.Interleaver] installs a controller on each of the model's modules.
  As the forward pass reaches each location, the controller calls
  [`Interleaver.handle(location, value)`][nnsight.intervention.interleaver.Interleaver.handle], which offers
  the value to the workers and caches interested in that location. A worker
  waiting on it is served the value (read) or has its replacement substituted in
  (swap); the possibly edited value is returned back into the model's execution.

Because a worker and the model take strict turns on one thread, there are no
locks or queues — only greenlet switches. Each [`Mediator`][nnsight.intervention.interleaver.Mediator] holds at most
one pending event at a time (the location it is currently parked on). A worker
must request locations in the order the model reaches them; asking for a
location the model already ran past raises [`OutOfOrderError`][nnsight.intervention.interleaver.OutOfOrderError].
"""

from __future__ import annotations

import enum
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

from greenlet import getcurrent, greenlet

from ..tracing.util import Scope

if TYPE_CHECKING:
    from .envoy import Envoy
    from .fragments import Fragments


class Event(enum.Enum):
    """What a parked worker is asking for.

    A worker parks by switching a tuple ``(Event, location, ...)`` to its parent;
    [`Mediator.handle`][nnsight.intervention.interleaver.Mediator.handle] inspects the first element to decide how to serve it.
    See [`Mediator.value`][nnsight.intervention.interleaver.Mediator.value], [`Mediator.swap`][nnsight.intervention.interleaver.Mediator.swap], [`Mediator.skip`][nnsight.intervention.interleaver.Mediator.skip], and
    [`Mediator.barrier`][nnsight.intervention.interleaver.Mediator.barrier] for how each is raised from intervention code.

    ``BARRIER`` is the odd one: it names no location, so the model side never
    serves it — another worker does, on its way past the same barrier.
    """

    VALUE = "VALUE"  # read a location:  (Event.VALUE, location)
    SWAP = "SWAP"  # replace a location: (Event.SWAP, location, value)
    SKIP = "SKIP"  # skip a computation: (Event.SKIP, location, value)
    BARRIER = "BARRIER"  # wait for the other blocks: (Event.BARRIER, None)


class Pending(NamedTuple):
    """What a worker is parked on, waiting for the model to reach.

    The occurrence is its own field so that matching a visit is a plain
    comparison; printing rejoins them (``'model.layers.16.output.i2'``), which is
    the form worth reading in an error.

    Attributes:
        event: What the worker wants done at ``provider`` — see
            [`Event`][nnsight.intervention.interleaver.Event].
        provider: The location, undecorated, or ``None`` for a barrier, which
            names no location and so is never served by the model side.
        iteration: Which occurrence of ``provider`` the worker is waiting for —
            the model has to have reached it this many times already.
        value: The replacement a swap or skip carries; unused by a read.
    """

    event: "Event"
    provider: Optional[str]
    iteration: Optional[int] = None
    value: Any = None

    def __str__(self) -> str:
        return f"{self.provider}.i{self.iteration}"


class EarlyStopException(Exception):
    """Raised by an intervention to halt the model run early.

    Thrown into the model's execution (e.g. via ``tracer.stop()``) to unwind the
    forward pass immediately. `Interleaver.__exit__` swallows it, since the
    early stop was intentional rather than a genuine error.
    """


class OutOfOrderError(Exception):
    """An intervention requested a location the model already ran past.

    Workers must ask for locations in the order the model reaches them. If the
    run finishes (or moves past a location) while a worker is still parked
    waiting for it, [`Interleaver.check_dangling_mediators`][nnsight.intervention.interleaver.Interleaver.check_dangling_mediators] throws this into
    the worker so the traceback points at the exact line that was waiting.
    """


class Mediator:
    """Runs one block of intervention code as a greenlet, in step with the model.

    A mediator wraps one captured block — the body of a ``with`` block, or one
    registered edit — and runs it inside a greenlet, the "worker". The worker
    drives the interaction: it runs until the intervention asks for a value, then
    parks, recording that pending request in [`pending`][nnsight.intervention.interleaver.Mediator.pending] and switching control
    back to the parent greenlet (the model side). The parent later resumes it
    through [`switch`][nnsight.intervention.interleaver.Mediator.switch] / `handle`.

    The classmethods [`value`][nnsight.intervention.interleaver.Mediator.value], [`swap`][nnsight.intervention.interleaver.Mediator.swap], `skip`, and
    `barrier` are the API the intervention code calls to park
    ([`Envoy`][nnsight.intervention.envoy.Envoy] properties like ``.output`` and ``.input`` are thin wrappers
    over them). `start`, [`switch`][nnsight.intervention.interleaver.Mediator.switch], and `handle` are the
    parent-side machinery that runs and feeds the worker. [`current`][nnsight.intervention.interleaver.Mediator.current] is how
    code inside a worker finds the mediator driving it.

    The block and its scope travel; everything the run builds does not — see
    `__getstate__`, which is how an edit rides to a remote server.

    Attributes:
        code: The captured block, compiled. Executed by the worker.
        glbls: The globals the block was written against.
        lcls: The [`Scope`][nnsight.tracing.util.Scope] the block runs in — its
            capture-time names, the frame it shares with the blocks written beside
            it, and those globals behind them. Doubles as what [`push_result`][nnsight.tracing.tracer.push_result]
            reads the block's results back out of.
        copy: Whether to exec against a fresh copy of [`lcls`][nnsight.intervention.interleaver.Mediator.lcls] each run. Set
            for an edit, which is replayed on every later trace and so must not
            accumulate the last replay's names.
        node: The block's AST node, kept so the mediator can serialize. ``None``
            for a mediator rebuilt server-side from already-reduced source.
        interleaver: The run this worker belongs to, set in `start`. Its
            ``batcher`` owns the row scoping `handle` applies.
        batch_group: This worker's ``[start, size]`` row range in the combined
            batch, or ``None`` for a whole-batch worker — an edit, or an empty
            invoke.
        worker: The greenlet running `code`, or ``None`` before
            `start`. Falsy once the worker has finished (see [`alive`][nnsight.intervention.interleaver.Mediator.alive]).
        pending: What the worker is currently parked on — a
            [`Pending`][nnsight.intervention.interleaver.Pending] naming the event,
            the location and which occurrence of it (see [`event`][nnsight.intervention.interleaver.Mediator.event]) — or ``None``
            when the worker isn't parked (before start or after it finishes).
        iteration: Which occurrence of a location the worker currently wants —
            the occurrence its pending request is matched under.
            ``tracer.iter`` pins it to a step; ``None`` means *relaxed* — the
            request resolves to the mediator's current count for that location (the
            next occurrence the model hasn't handled). Stays ``0`` (the first
            occurrence) with no ``tracer.iter``. Relaxes to ``None`` after the
            first hit of a pinned non-zero step (see `handle`).
        base: The interleaver's per-location counts when this worker started;
            its own occurrence of a location is the interleaver's count minus this
            (see [`occurrence`][nnsight.intervention.interleaver.Mediator.occurrence]).
        caches: The caches this worker's ``tracer.cache()`` created. They observe
            every location the run reaches, after interventions have had it.
    """

    def __init__(
        self,
        code: Any,
        glbls: dict,
        lcls: dict,
        copy: bool = False,
        node: Any = None,
        shared: dict | None = None,
    ) -> None:
        # The captured block to run and the scope it runs against (see
        # [`Scope`][nnsight.tracing.util.Scope] for how a block reaches names).
        # ``shared`` is the live locals of the frame the block was written in, so
        # blocks written together see each other's binds; a block with no siblings
        # (a deserialized edit, replayed with no frame at all) shares nothing.
        # ``copy`` (edits, which are stored and replayed on every future trace)
        # execs against a fresh copy each run so a replay doesn't accumulate the
        # block's own mutations; otherwise (a one-shot trace/invoke body) it execs
        # against the stored scope so push_result can read back the values it saved.
        self.code = code
        self.glbls = glbls
        self.lcls = Scope(lcls, {} if shared is None else shared, glbls)
        self.copy = copy
        # The block's AST node, kept so the mediator can serialize (edits ride with
        # the model to a remote server): __getstate__ reduces it to source + the vars
        # it references, exactly like the traced block. None for a mediator built
        # server-side from already-reduced source (it isn't re-serialized).
        self.node = node
        # Batch scoping (see intervention/batching.py): the interleaver this worker
        # runs under (its `batcher` owns the narrow/widen logic), and this worker's
        # row range in the combined batch. `interleaver` is set in `start`;
        # `batch_group` is None for a whole-batch worker (an edit or empty invoke).
        self.interleaver: Any = None
        self.batch_group: list | None = None
        self.worker: greenlet | None = None
        # What the worker is currently parked on waiting to be served (a
        # [`Pending`][nnsight.intervention.interleaver.Pending]), or None when it
        # isn't parked.
        self.pending: Pending | None = None
        # Which occurrence of a location the worker is currently asking for (or
        # None when relaxed). See `handle` for how it is matched.
        self.iteration: int | None = 0
        # Whether the `tracer.iter` loop moving that pin has an end (set by
        # Iterations.__iter__). A loop with an end that outruns the run is an
        # error rather than the warning an open one gets; see
        # `Interleaver.check_dangling_mediators`.
        # The interleaver's counts when this worker started; its own occurrence
        # of a location is the interleaver's count minus this.
        self.counts_at_start: dict[str, int] = {}
        # Caches created by this worker's `tracer.cache()`. They observe every
        # location this run reaches (post-intervention); see Interleaver.handle.
        self.caches: list = []
        # A one-shot write-back bound by an eproperty read whose value was a
        # reshaped view (see eproperty.transform): fired once, right after this
        # worker's read on a location, to splice the edited view back in `handle`.
        self.transform: Optional[Callable] = None
        # The exception this worker raised, if any — a tracer.stop()'s
        # EarlyStopException or a genuine error in intervention code. Set only under
        # a deferring interleaver (see Interleaver.defer_exceptions): a driver that
        # keeps running past a worker's error — one whose forward is a step of a
        # run it schedules itself — reads this to end that worker's run and, for a
        # real error, carry it back to the trace that wrote it.
        self.exception: Optional[BaseException] = None
        # Names whose values were `.save()`d in the process that serialized
        # this worker (see __getstate__); a receiving driver unions them into
        # its own saved-name collection.
        self.presaved: set[str] = set()

    def _run(self) -> None:
        """Execute the captured block (the worker greenlet's body).

        The scope is ``exec``'s globals as well as its locals, so a ``lambda`` or
        nested ``def`` in the block can reach the block's own names (see
        [`Scope`][nnsight.tracing.util.Scope]).
        """
        exec(self.code, self.lcls.copy() if self.copy else self.lcls)

    def __getstate__(self) -> dict:
        # Ships with the model (an edit rides in envoy._edits to a remote server).
        # Reduce the block to source + the vars it references — cross-version safe,
        # exactly like the traced block — and drop the compiled code and all run
        # state (worker/pending/interleaver/batch_group can't and shouldn't travel).
        from ..tracing.tracer import _saves
        from .serialization import reduce_block

        reduced = reduce_block(self.node, self.glbls, self.lcls)
        # Saved-ness travels by NAME: `.save()` marks object ids, which are
        # local to this process, so the receiving driver gets the names of the
        # captured variables that were saved (e.g. a container bound and saved
        # above the invoke blocks).
        presaved = {
            name for name, value in reduced[2].items() if id(value) in _saves()
        }
        # Kept on this side too: it is the only record of which names were bound
        # (and saved) *above* the block, which is what tells a value shared by
        # every request apart from one each request computed for itself.
        self.presaved = presaved
        return {"reduced": reduced, "copy": self.copy, "presaved": presaved}

    def __setstate__(self, state: dict) -> None:
        import linecache

        source, glbls, lcls = state["reduced"]
        # Register the block's source under a unique filename so source
        # introspection works on the receiving side: a nested
        # ``with tensor.backward():`` captures its body from linecache, and
        # tracebacks resolve to real lines.
        filename = f"<edit-{id(self)}>"
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        self.__init__(compile(source, filename, "exec"), glbls, lcls, copy=state["copy"])
        self.presaved = state["presaved"]

    @classmethod
    def current(cls, what: str) -> "Mediator":
        """The mediator whose worker is running now.

        Only intervention code has one, and intervention code only runs while
        interleaving — so no worker means ``what`` was asked for outside a run,
        and there is nothing to park on and nothing to answer with.

        Raised as a `ValueError` rather than the ``AttributeError`` that
        reaching for the absent worker gives: from a property like
        [`output`][nnsight.intervention.envoy.Envoy.output], an ``AttributeError`` is
        taken for "no such attribute" and comes back out of ``__getattr__`` as one,
        naming the property instead of the reason.
        """
        mediator = getattr(getcurrent(), "mediator", None)
        if mediator is None:
            raise ValueError(f"Cannot access `{what}` outside of interleaving")
        return mediator()

    @classmethod
    def event(cls, event: Event, location: str, *rest: Any) -> Any:
        """Raise an event from inside a worker and return what's sent back.

        Called on the *worker* side (from intervention code): switch to the parent
        greenlet — the interleaver driving the model — handing it the event tuple,
        and block until the parent switches a value back in. This is the
        counterpart to [`switch`][nnsight.intervention.interleaver.Mediator.switch], which drives the worker from the parent.

        ``location`` is tagged ``.i{n}`` with the occurrence the worker wants, so
        `handle` can bind it with a single match. When pinned
        ([`iteration`][nnsight.intervention.interleaver.Mediator.iteration] is an int), that's the pinned step. When relaxed
        (``None``), it's the mediator's current count for this location — the next
        occurrence the model hasn't handled yet — so the request follows the model
        sequentially.
        """
        mediator = cls.current(location)
        worker = getcurrent()
        iteration = (
            mediator.iteration
            if mediator.iteration is not None
            else mediator.occurrence(location)
        )
        return worker.parent.switch(Pending(event, location, iteration, *rest))

    def occurrence(self, location: str) -> int:
        """How many times the model has reached ``location`` since this worker started."""
        counts = {} if self.interleaver is None else self.interleaver.counts
        return counts.get(location, 0) - self.counts_at_start.get(location, 0)


    @classmethod
    def value(cls, location: str) -> Any:
        """Read the value at ``location`` from inside a worker.

        Parks until the interleaver reaches ``location`` — a module input/output
        path (e.g. ``"model.h.0.output"``) or the run's ``"result"`` — then returns
        the value produced there.
        """
        return cls.event(Event.VALUE, location)

    @classmethod
    def swap(cls, location: str, value: Any) -> None:
        """Replace the value at ``location`` from inside a worker.

        Parks like [`value`][nnsight.intervention.interleaver.Mediator.value], but when the interleaver reaches ``location`` it
        substitutes ``value`` for what the model produced (see `handle`), then
        resumes the worker. Reading then swapping the same location works — both
        events are drained in one `handle`.
        """
        cls.event(Event.SWAP, location, value)

    @classmethod
    def skip(cls, location: str, value: Any) -> None:
        """Skip the computation gated at ``location``, using ``value`` as its result.

        Parks like [`swap`][nnsight.intervention.interleaver.Mediator.swap], but targets a ``.skip`` gate that a module's (or
        op's) forward wrapper queries *before* running — so ``value`` is returned
        in place of running the computation, not after. A distinct event from SWAP
        so skip-specific behavior can hang off it later.
        """
        cls.event(Event.SKIP, location, value)

    @classmethod
    def barrier(cls) -> None:
        """Park until another block releases this worker.

        Unlike [`value`][nnsight.intervention.interleaver.Mediator.value] / [`swap`][nnsight.intervention.interleaver.Mediator.swap] / `skip`, this parks on nothing
        the model produces: it waits on the other *blocks*, and the last of them
        to arrive is what resumes it (see
        [`Barrier`][nnsight.intervention.barrier.Barrier]). Its pending event carries
        no location, so the model side never serves it.
        """
        getcurrent().parent.switch(Pending(Event.BARRIER, None))

    @property
    def alive(self) -> bool:
        """Whether the worker exists and still has intervention code left to run.

        ``False`` before `start` (no worker yet) and after the worker
        finishes — a greenlet is falsy once it has run to completion — so this is
        only ``True`` while the worker is parked mid-intervention.
        """
        return bool(self.worker)

    def start(self, interleaver: "Interleaver" = None) -> None:
        """Create the worker greenlet and run it up to its first park.

        Switching into a fresh greenlet runs the captured block until it first
        parks on a location (or finishes). Whatever it parks on becomes
        [`pending`][nnsight.intervention.interleaver.Mediator.pending], ready for `handle` to serve once the model reaches
        that location. Per-run counters are reset here so a stored edit mediator,
        replayed on a later trace, starts clean. ``interleaver`` is the run this
        worker belongs to (it reads batch scoping from ``interleaver.batcher``).
        """
        self.interleaver = interleaver
        self.iteration = 0
        self.counts_at_start = dict(interleaver.counts) if interleaver is not None else {}
        self.caches = []
        self.transform = None
        self.worker = greenlet(run=self._run)
        # Let intervention code reach its own mediator from inside the worker via
        # getcurrent().mediator() — to tag a park with the current iteration
        # ([`event`][nnsight.intervention.interleaver.Mediator.event]) or move it (`tracer.iter`). A weakref so the worker
        # doesn't hold the mediator (which holds the worker) alive in a cycle.
        self.worker.mediator = weakref.ref(self)
        self.pending = self.switch()

    def switch(self, *args: Any) -> Any:
        """Resume the worker with ``args``; return the next event it parks on.

        Switches control into the worker greenlet, handing it ``args`` as the
        return value of whatever park call it was blocked in, and blocks until
        the worker parks again (returning its new event tuple) or finishes
        (returning ``None``). If the worker raises, its traceback is stashed on
        the exception as ``__intervention_tb__`` — a clean, intervention-only
        trace captured before the re-raise unwinds the model's own stack on top —
        and the exception propagates, halting the run.

        Re-point the worker's parent at whoever is switching in *now*, so both its
        return paths — parking (``worker.parent.switch(...)``) and finishing (a
        greenlet auto-returns to its parent) — go back here rather than to a fixed
        greenlet. This keeps the chain correct when a worker is served from inside
        another worker's greenlet, e.g. an ``Envoy.__call__(hook=True)`` adapter
        run whose submodule controllers serve a second worker mid-call.
        """
        self.worker.parent = getcurrent()
        try:
            return self.worker.switch(*args)
        except BaseException as exception:
            # The worker raised. Its traceback right now holds only the
            # intervention frames; stash it before the re-raise unwinds the
            # model stack on top, so the top can restore a clean trace.
            # The exception still propagates, stopping execution immediately.
            if not hasattr(exception, "__intervention_tb__"):
                exception.__intervention_tb__ = exception.__traceback__
            raise

    def handle(self, provider: str, value: Any) -> Any:
        """Drain the worker's events parked on this visit to ``provider``; return the value.

        A read ([`Event.VALUE`][nnsight.intervention.interleaver.Event.VALUE]) is served ``value``; a swap
        ([`Event.SWAP`][nnsight.intervention.interleaver.Event.SWAP]) replaces ``value`` with the worker's. The worker may
        do both in turn (read a location, then assign it), so loop until it parks
        somewhere else or finishes. The returned value flows back up through
        [`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle] to the controller, which substitutes it into the run.

        A location can be reached many times in one run — e.g. a module revisited
        on every step of a generation loop. This visit is the
        `occurrence(provider)`-th, so it serves the workers waiting for that
        occurrence of it. A worker parks already carrying the occurrence it wants
        (see [`event`][nnsight.intervention.interleaver.Mediator.event]) — pinned to a step, or resolved to the next
        occurrence when relaxed — so this is a location match and an integer
        match: a request pinned to a later step doesn't match yet and waits while
        earlier visits pass by. With no ``tracer.iter`` the occurrence is always
        ``0``, so every request binds to the first visit. Once a pinned non-zero
        step is hit, the mediator is relaxed to ``None`` so the rest of that
        step's requests follow the model sequentially rather than re-forcing the
        index.
        """
        pending = self.pending
        if pending is None or pending.provider != provider:
            return value

        iteration = self.occurrence(provider)
        # The batcher (if this run is batching) scopes a value to this worker's rows.
        batcher = None if self.interleaver is None else self.interleaver.batcher
        while (
            pending is not None
            and pending.provider == provider
            and pending.iteration == iteration
        ):
            # First hit of an explicit iter[n] (n > 0): drop the pin so the
            # remaining requests this step follow the model sequentially (event
            # resolves them to the current count). 0 and None don't relax (0 is
            # "unpinned"; None already is).
            if self.iteration:
                self.iteration = None
            if pending.event is Event.VALUE:  # serve this worker only its rows
                served = value if batcher is None else batcher.narrow(value, self.batch_group)
                self.pending = self.switch(served)
                # If that read bound a write-back (an eproperty whose preprocess
                # returned a view), the worker has since edited the view — fire it
                # and splice the mapped-back result in, exactly like a swap.
                if self.transform is not None:
                    mapped = self.transform()
                    value = (
                        mapped
                        if batcher is None
                        else batcher.widen(value, self.batch_group, mapped)
                    )
                    self.transform = None
            elif pending.event is Event.SWAP:  # splice its edit back into the batch
                if batcher is None:
                    value = pending.value
                else:
                    value = batcher.widen(value, self.batch_group, pending.value)
                self.pending = self.switch()
            elif pending.event is Event.SKIP:  # gather per-invoke replacements
                if batcher is None:
                    value = pending.value
                else:
                    value = batcher.gather_skip(
                        value, self.batch_group, pending.value
                    )
                self.pending = self.switch()
            pending = self.pending
        return value


def dangling_unwind(mediator: "Mediator") -> tuple[BaseException, Optional[str]]:
    """How a worker still parked when its run is over should be ended.

    Returns the error to throw into the worker — the throw unwinds it, running its
    ``finally`` blocks, and points the traceback at the line that was waiting — and
    the warning to emit *instead of* surfacing that error, or ``None`` when the
    error stands. Two drivers ask, and they differ only in what surfacing means:
    [`Interleaver.check_dangling_mediators`][nnsight.intervention.interleaver.Interleaver.check_dangling_mediators]
    raises the error, and vLLM's ``Requests.finish_dangling`` carries it home as the
    request's deferred error. The policy is one copy here because two copies of it
    can disagree, and a block that is an error locally and a warning on the engine
    is a block whose meaning depends on where it runs.

    Two cases:

    * **Out of order** — a plain request (``iteration == 0``) for a location the
      model already ran past, or never called. A real error.
    * **A loop that outran the run** — ``tracer.iter[...]`` asked for a step the
      run did not make. An open ``tracer.iter[:]`` / ``tracer.all()`` has no end of
      its own, so outrunning the model is how it finishes; a bounded loop cut short
      by an EOS or a stop string ends the same way. That one warns: values from the
      steps that were reached are already saved, and the statements after the loop
      do not run.

    Call this *before* the throw: unwinding the worker's loop restores both the pin
    and the loop's shape it reads.
    """
    requester = mediator.pending
    if requester.event is Event.BARRIER:
        # Fewer blocks reached the barrier than it was built for, so it was never
        # going to release.
        return (
            ValueError(
                "A barrier was never reached by every block it waits for; "
                "check the count it was created with"
            ),
            None,
        )
    if mediator.iteration == 0:
        return (
            OutOfOrderError(
                f"'{requester}' was requested but the model already ran past it"
            ),
            None,
        )
    return (
        OutOfOrderError(
            f"'{requester}' was requested but the model already ran past it"
        ),
        f"'{requester}' was never reached: the loop asked for a step the run "
        f"did not make, so it was cut short — values saved inside the loop "
        f"are kept, and the statements after it did not run. An open "
        f"`tracer.iter[:]` / `tracer.all()` loop ends this way by design. To "
        f"hold a generation to a bounded loop's count, pass `min_new_tokens=` "
        f"on transformers or `min_tokens=` / `ignore_eos=True` on vLLM; put "
        f"what follows the loop in a separate `tracer.invoke()`.",
    )


class Interleaver:
    """Drives the model side of interleaving: model handoffs in, workers served.

    An interleaver owns the module controllers that turn a model's forward pass
    into a stream of `handle` calls, and the list of [`Mediator`][nnsight.intervention.interleaver.Mediator] workers
    those calls feed. One interleaver is shared across an [`Envoy`][nnsight.intervention.envoy.Envoy] tree, so
    every module's controller reports into the same set of workers.

    Lifecycle of a run (see [`Envoy.interleave`][nnsight.intervention.envoy.Envoy.interleave]):

    1. A [`Mediator`][nnsight.intervention.interleaver.Mediator] is appended to `mediators` for each intervention
       block and each registered edit.
    2. Entering the interleaver (``with interleaver:``) flips
       [`interleaving`][nnsight.intervention.interleaver.Interleaver.interleaving] on and [`start`][nnsight.intervention.interleaver.Mediator.start]\\ s every worker so each
       parks on its first requested location.
    3. The model runs. Each module's controller (see [`instrument`][nnsight.intervention.interleaver.Interleaver.instrument]) calls
       `handle`, serving reads and applying swaps for any worker parked
       there, and returns the (possibly edited) value into the forward pass.
    4. [`check_dangling_mediators`][nnsight.intervention.interleaver.Interleaver.check_dangling_mediators] surfaces any worker still waiting for a
       location the model never reached ([`OutOfOrderError`][nnsight.intervention.interleaver.OutOfOrderError]), and
       [`cancel`][nnsight.intervention.interleaver.Interleaver.cancel] clears the workers so the next run starts clean.

    Attributes:
        mediators: The workers to serve this run.
        batcher: The [`Batcher`][nnsight.intervention.batching.Batcher] for this run,
            which assembled the combined input and owns the row scoping
            [`Mediator.handle`][nnsight.intervention.interleaver.Mediator.handle] applies — or ``None`` when not batching.
            Cleared by [`cancel`][nnsight.intervention.interleaver.Interleaver.cancel].
        interleaving: ``True`` between ``__enter__`` and ``__exit__``. Hooks pass
            values straight through when it is ``False``, so an instrumented
            model runs normally outside a trace.
        sourced: Op-location -> the instrumented callable a worker drilled into
            (see [`nnsight.intervention.source`][nnsight.intervention.source]), or ``None`` while one is
            requested but not yet built. Per-run; cleared on entry.
        fragments: A [`Fragments`][nnsight.intervention.fragments.Fragments] for a
            model whose values are split across devices, or ``None``. When set,
            `handle` gathers a fragment before serving workers and re-splits it
            afterwards.
    """

    def __init__(self, fragments: Optional["Fragments"] = None) -> None:
        # The workers this run serves. Local tracing appends to it before the run
        # is entered, and `__enter__` indexes it; a driver that swaps the list
        # while a run is in progress calls `reindex` itself afterwards.
        self.mediators: list[Mediator] = []
        # The envoy wrapping each module of this tree, by module id, so a module
        # reachable by two paths gets one envoy (the first path) and an alias.
        # Weak: the tree's parent links own the envoys; this only looks them up.
        self.envoys: "weakref.WeakValueDictionary[int, Any]" = weakref.WeakValueDictionary()
        # The locations some worker is parked on, so a visit with nobody waiting
        # costs one set lookup. Rebuilt from the workers by `reindex` at the run
        # boundaries and by `handle` after it has served anyone (the only time a
        # worker can park outside those).
        self.parked: set[str] = set()
        # How many times each location has been handled, for the life of the
        # tree. A worker's own count of a location is this minus what it stood at
        # when the worker started (`Mediator.occurrence`), so one increment per
        # visit serves every worker however many are live.
        self.counts: dict[str, int] = {}
        # Location -> the caches that keep it, as (worker, cache, what to store).
        self.observers: dict[str, list[tuple[Mediator, Any, tuple[str, str]]]] = {}
        # What, if anything, makes a value at a location whole before workers see
        # it — see intervention/fragments.py. None on an ordinary model, and on a
        # distributed one it is the runtime's own object. Kept as a collaborator
        # rather than a subclass because *when* to gather is a property of this
        # class and *what* to gather is a property of the runtime.
        self.fragments: Optional["Fragments"] = fragments
        # The batcher for the current run (combined-input assembly + narrow/widen),
        # or None when not batching. Owned by the tracer and registered here for the
        # run by `Envoy.interleave` (see intervention/envoy.py); each mediator reads
        # it for its row scoping. Cleared by cancel().
        self.batcher: Any = None
        self.interleaving = False
        # Whether a module's handoff has anything to reach: recomputed with the
        # worker list (`reindex`), and the gate every module checks before handing
        # off, so a run with no workers costs a module one attribute test. A
        # runtime that needs the handoff to run with no workers present sets it
        # itself.
        self.busy = False
        # Recursive source (see intervention/source.py). Maps an op-location path a
        # worker asked to drill into to its instrumented callable + Compiled, or to
        # `None` while requested-but-not-yet-built: `None` tells the model side to
        # hand the live callable to the parked worker over a `{path}.fn` location;
        # the built entry is then reused by later fires this run (e.g. generation
        # steps). Per-run — cleared on entry.
        self.sourced: dict[str, tuple | None] = {}
        # When True, a worker's exception is caught and recorded on its mediator
        # rather than raised out of the handoff. A driver whose forward is one step
        # of a run shared by several workers sets this so one worker's error ends
        # only its own run instead of tearing down the others'.
        self.defer_exceptions = False

    def reindex(self) -> None:
        """Rebuild the parked set, cache routes and `busy` from `mediators`.

        Called by `__enter__` once the workers have started, and by a driver that
        replaces the list while a run is in progress (a scheduler reshuffling
        which workers are in the batch), since nothing else notices the swap.
        """
        self.observers = {}
        self.busy = bool(self.mediators)
        self._reindex_parked()

        for mediator in self.mediators:
            for cache in mediator.caches:
                for provider, selected in cache.subscriptions().items():
                    self.observers.setdefault(provider, []).append((mediator, cache, selected))

    def _reindex_parked(self) -> None:
        self.parked = {
            pending.provider
            for mediator in self.mediators
            if (pending := mediator.pending) is not None and pending.provider is not None
        }

    def _ready(self, provider: str) -> list[Mediator]:
        """The workers parked on this visit of ``provider``."""
        return [
            mediator
            for mediator in self.mediators
            if (pending := mediator.pending) is not None
            and pending.provider == provider
            and pending.iteration == mediator.occurrence(provider)
        ]

    def __enter__(self) -> Interleaver:
        """Begin interleaving: arm the controllers and start each not-yet-started worker.

        Only a worker with no greenlet yet (``worker is None``) is started; one that
        already has a worker is left as is — parked mid-run on a re-entered interleaver.
        The gate tests ``worker`` rather than [`alive`][nnsight.intervention.interleaver.Mediator.alive] so that a worker
        whose block has *finished* is also left alone: a finished greenlet is falsy, so
        an ``alive`` gate would take it for never-started and rerun its whole block.
        """
        self.interleaving = True
        self.sourced.clear()
        try:
            for mediator in self.mediators:
                if mediator.worker is not None:
                    continue
                mediator.start(self)
        except BaseException:
            # A worker that errors on start (e.g. invoking mid-run) means __exit__
            # won't run to clear the flag, so reset it here or it leaks to the next run.
            self.interleaving = False
            raise
        # After the workers have parked, so this run's index describes this run's
        # workers — including any registered by appending to the list rather than
        # replacing it, which is how the local path does it.
        self.reindex()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """End interleaving, swallowing an intentional early stop.

        Returning ``True`` for [`EarlyStopException`][nnsight.intervention.interleaver.EarlyStopException] suppresses it: an
        intervention asked to halt the run and has already unwound the model, so
        it is not an error. Any other exception propagates.
        """
        self.interleaving = False
        # An EarlyStopException means an intervention asked to halt the run; it
        # has done its job unwinding the model, so swallow it here.
        return exc_type is EarlyStopException

    def instrument(self, envoy: Envoy) -> None:
        """Route an envoy's module through this interleaver.

        Installs the module's controller (see
        [`install_controller`][nnsight.intervention.source.install_controller]), which
        hands the module's input and output to `handle` under the locations
        ``"{path}.input"`` and ``"{path}.output"`` while
        [`interleaving`][nnsight.intervention.interleaver.Interleaver.interleaving], and gates
        ``.skip``. No forward hooks: a module with none is called on PyTorch's
        fast path, which is what keeps an instrumented model's cost near zero
        outside a trace. Also registers this interleaver on the module, so a
        module can be skipped or source-drilled by this trace — and by another
        envoy sharing the module at the same time.
        """
        from .source import install_controller  # lazy: source imports this module

        # The one moment both the module — carrying whatever its runtime stamped
        # on it — and its path are in hand, which is what a distributed runtime
        # needs to record which of this envoy's values are pieces of larger ones.
        # Called again through `Envoy._update` when real weights are dispatched
        # under a tree built on meta, which is where a shard first becomes
        # visible.
        if self.fragments is not None:
            self.fragments.instrument(envoy)

        install_controller(envoy)

    def handle(self, provider: str, value: Any) -> Any:
        """Route ``value`` to this provider's consumers; return it, edited if any
        intervention wrote to this location.

        The provider index picks only workers parked here and caches that selected
        it. The visit is counted once, on the interleaver, after everyone parked on
        it has been served — so a worker that arrives here *during* the visit
        (released from a barrier by one that was served) asks for this occurrence
        and is served in it too, whichever order the workers were written in.
        """
        occurrence = self.counts.get(provider, 0)
        observers = (
            self.observers.get(provider, ())
        )
        # The common path: nobody parked here and no cache observing.
        if provider not in self.parked and not observers:
            self.counts[provider] = occurrence + 1
            return value

        # A fragment is made whole before any consumer sees it, but only when a
        # worker will be served on *this* occurrence or a selected cache records
        # it. Every rank derives this from the same provider routes, keeping their
        # collectives matched.
        undo = None
        if self.fragments is not None and self.fragments.enabled and self.fragments.fragmented(provider) and (observers or self._ready(provider)):
            value, undo = self.fragments.whole(provider, value)

        # Serving one worker can release another into parking here (a barrier),
        # so keep serving until nobody is parked on this visit.
        served = False
        while ready := self._ready(provider):
            served = True
            for mediator in ready:
                try:
                    value = mediator.handle(provider, value)
                except Exception as exception:
                    if not self.defer_exceptions:
                        raise
                    mediator.exception = exception
                    mediator.pending = None
        if served:
            self._reindex_parked()
        self.counts[provider] = occurrence + 1

        # A batched skip leaves its invokes' replacements gathered rather than one
        # value; concatenate them into the combined output (see
        # Batcher.gather_skip). A future-iteration waiter has no gathered skip,
        # but retaining this gate preserves the previous provider-level behaviour.
        if self.batcher is not None:
            value = self.batcher.assemble_skip(value)

        for mediator, cache, selected in observers:
            served = (
                value
                if self.batcher is None
                else self.batcher.narrow(value, mediator.batch_group)
            )
            cache.observe_selected(selected, served)

        # Back to the piece the model's own forward expects, carrying whatever the
        # workers left behind — so an edit to the assembled tensor reaches the
        # model rather than being dropped with the gather.
        if undo is not None:
            value = undo(value)
        return value

    def check_dangling_mediators(self) -> None:
        """Surface any worker still parked after the run.

        Called once the model has finished. A worker that is still [`alive`][nnsight.intervention.interleaver.Mediator.alive]
        was waiting for a location the model never reached.
        [`dangling_unwind`][nnsight.intervention.interleaver.dangling_unwind] decides
        what that means — an out-of-order read, a bounded ``tracer.iter`` loop the
        run could not supply, or an open one ending the only way it can — and this
        is the local half of it: what stands is raised, out of the worker's own
        frame, and what is expected is warned about.
        """
        for mediator in self.mediators:
            if not mediator.alive:
                continue
            error, expected = dangling_unwind(mediator)
            try:
                mediator.worker.throw(error)
            except BaseException as thrown:
                # `thrown is error` only when the block let the unwind through: a
                # `finally` that raises something of its own is that error, not a
                # loop that outran the run, and stands however the loop was written.
                if expected is None or thrown is not error:
                    raise
                warnings.warn(expected)

    def cancel(self) -> None:
        """Drop all mediators and the batcher so the next run starts clean.

        Each mediator's worker greenlet is released too, so a stored edit mediator
        replayed on a later trace is seen as never-started (``worker is None``) and
        restarts fresh rather than being skipped for still holding its finished
        greenlet. Surfacing dangling mediators is a separate concern (see
        [`check_dangling_mediators`][nnsight.intervention.interleaver.Interleaver.check_dangling_mediators]), handled by the driver after a run.
        """
        for mediator in self.mediators:
            mediator.worker = None
        self.mediators = []
        self.batcher = None


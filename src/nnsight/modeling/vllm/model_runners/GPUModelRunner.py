"""Run interventions inside vLLM's worker, against the real weights.

This is where a trace written in another process actually happens. The runner
builds its own [`VLLM`][nnsight.modeling.vllm.vllm.VLLM] over the module vLLM
loaded, so the module tree here has the same paths the client wrote against; a
worker arriving on a request then resolves straight onto the real modules.

Three points on vLLM's own path carry it:

* ``_update_states`` — the scheduler has just decided what runs this step, so new
  requests hand over their workers and every worker's token span is recomputed.
* ``execute_model`` — the forward, run with the interleaver open so hooks serve
  the parked workers.
* ``sample_tokens`` / ``_sample`` — logits and sampled ids never pass through a
  module, so they are offered to workers directly by location.

Workers run as greenlets on this thread. ``_update_states`` is called from
``execute_model``, and hooks fire on whichever thread runs the forward, so the
worker and the model take strict turns on one thread — there is nothing to
synchronize *during the forward*. Collection (``collect_nnsight``) is the
exception: under Ray it lands on a different thread than the forward, which is why
saves and errors are snapshotted onto the mediator on the worker thread
(``record_saves``, ``finish_dangling``) rather than read live at collect time.
"""

from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from ....intervention.interleaver import Interleaver, Mediator
from ....intervention.serialization import loads
from ....tracing.tracer import _local, _saves, inc
from ..batching import VLLMBatcher
from ..fragments import VLLMFragments

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

    from ..vllm import VLLM


class Requests:
    """The workers riding this engine's in-flight requests.

    Two kinds of worker run here. A *traced* one arrives on its request, one per
    request, and goes home when that request finishes. A *registered* one is a
    block left on the engine ahead of time (see
    [`nnsight.modeling.vllm.registration`][nnsight.modeling.vllm.registration]):
    the template is deserialized once, and every request that arrives afterwards
    — whether or not it is an nnsight trace at all — gets a fresh copy with a
    scope of its own, whose saves are kept here until they are collected.

    Attributes:
        mediators: Traced worker by request id, for as long as the request lives.
        errored: Deferred error by request id for a payload that failed to
            deserialize — surfaced at collect instead of crashing the engine.
        rows: Request ids in the order the forward's tensors carry them.
        tokens: Token count each request contributes this step, by request id.
        templates: Registration id -> what a per-request copy of that block is
            built from, so the source is compiled once rather than per request.
        registered: Request id -> registration id -> that request's own copy.
        harvested: Registration id -> engine request id -> the values that
            request's copy saved, waiting to be collected.
    """

    def __init__(self) -> None:
        self.mediators: dict[str, Any] = {}
        self.errored: dict[str, Any] = {}
        self.rows: list[str] = []
        self.tokens: dict[str, int] = {}
        self.templates: dict[str, tuple] = {}
        self.registered: dict[str, dict[str, Any]] = {}
        self.harvested: dict[str, dict[str, dict]] = {}

    def register(self, registration_id: str, payload: bytes, persistent_objects: dict) -> None:
        """Take a block the engine should run for every request from now on.

        Deserializing recompiles the block's source, so it is done once here and
        the pieces kept; `add` then builds each request's copy from them, which
        is the whole point of registering rather than tracing.
        """
        template = loads(payload, persistent_objects=persistent_objects)
        self.templates[registration_id] = (
            template.code,
            template.glbls,
            dict(template.lcls),
            template.presaved,
        )
        self.harvested.setdefault(registration_id, {})

    def unregister(self, registration_id: str) -> None:
        """Stop running a block and forget anything it has not handed back."""
        self.templates.pop(registration_id, None)
        self.harvested.pop(registration_id, None)
        for per_request in self.registered.values():
            per_request.pop(registration_id, None)

    def add(
        self, new_requests: list["NewRequestData"], persistent_objects: dict
    ) -> None:
        """Take the worker off each new request that carries one.

        A request with no nnsight payload is another tenant of the same engine and
        is left alone — it still occupies tokens in the batch, so [`scope`][nnsight.modeling.vllm.model_runners.GPUModelRunner.Requests.scope]
        counts it, but nothing runs for it. Most steps (every decode step) bring no
        new requests, so this returns immediately then.

        A payload that fails to deserialize (a corrupt or version-mismatched request)
        is caught and recorded as this request's error rather than raised: it runs
        inside ``super().execute_model``, so letting it propagate would tear down the
        engine every other tenant shares. The error is surfaced at collect.
        """
        from ....intervention.errors import capture_exception

        for request in new_requests:
            # Registered blocks apply to every request, including ones carrying no
            # nnsight payload at all — that is what registering is for. Each gets
            # its own mediator so its scope, and so what it saves, is this
            # request's alone.
            if self.templates:
                copies = self.registered.setdefault(request.req_id, {})
                for registration_id, template in self.templates.items():
                    code, glbls, lcls, presaved = template
                    mediator = Mediator(code, glbls, dict(lcls))
                    mediator.presaved = set(presaved)
                    copies[registration_id] = mediator

            extra_args = getattr(request.sampling_params, "extra_args", None) or {}
            if "nnsight_mediator" not in extra_args:
                continue
            try:
                self.mediators[request.req_id] = loads(
                    extra_args["nnsight_mediator"],
                    persistent_objects=persistent_objects,
                )
            except Exception as exception:
                self.errored[request.req_id] = capture_exception(exception)

    def _by_row(self):
        """Each scheduled request's token count and its workers, in batch order.

        Registered copies come first, so a trace written for this request reads
        whatever they left behind rather than racing them. A request the scheduler
        did not run this step contributes no tokens and no row.
        """
        for request_id in self.rows:
            tokens = self.tokens.get(request_id)
            if tokens is None:
                continue
            workers = list(self.registered.get(request_id, {}).values())
            mediator = self.mediators.get(request_id)
            if mediator is not None:
                workers.append(mediator)
            yield tokens, workers

    def scope(self, model: "VLLM") -> None:
        """Point every worker at its own tokens within this step's batch.

        A worker's span is only meaningful for the step it was computed in: the
        scheduler decides which requests run and how many tokens each contributes,
        so a span from an earlier step would index into a batch that no longer
        exists. Workers whose request isn't running now are dropped from the
        interleaver and report no group, and a worker whose block already ran to
        completion is dropped too — the interleaver starts anything not alive, and
        a finished block must not be run a second time.
        """
        for mediator in self.mediators.values():
            mediator.batch_group = None
        for copies in self.registered.values():
            for mediator in copies.values():
                mediator.batch_group = None

        interleaver = model.interleaver
        scheduled = []
        start = 0

        def take(mediator: Any, tokens: int) -> None:
            """Give one worker this request's span, starting it the first time."""
            # A block that already finished is normally dropped — its work is
            # done and it must not run a second time. Two exceptions stay
            # scheduled (never restarted, only kept in view): one holding open
            # caches (tracer.cache()) keeps observing every step until the request
            # ends, and one that raised keeps a row here so _finish_erred can force
            # its end-of-sequence every step until vLLM actually retires it —
            # forcing once is not enough when min_tokens defers the stop.
            # A mediator has no worker until start(); once started it keeps a
            # (dead once its block finishes) greenlet. So `worker is not None`
            # means started, and started-but-not-alive means the block finished.
            started = mediator.worker is not None
            finished = started and not mediator.alive
            if finished and not mediator.caches and mediator.exception is None:
                return
            mediator.batch_group = [start, tokens]
            scheduled.append(mediator)
            # Which requests run is only settled here, once the forward has
            # already begun, so a worker is started the moment its request
            # is first scheduled rather than on the way into the interleaver.
            if not started:
                try:
                    mediator.start(interleaver)
                except Exception as exception:
                    # A block that errors before it first parks (a bad line
                    # at the top) is deferred here, like one that errors
                    # mid-run in the interleaver; _finish_erred ends it.
                    if not interleaver.defer_exceptions:
                        raise
                    mediator.exception = exception

        for tokens, workers in self._by_row():
            for mediator in workers:
                take(mediator, tokens)
            start += tokens

        # Every scheduled request's tokens, nnsight's or not — the leading dim of
        # the activations a worker will be narrowed out of.
        interleaver.batcher.total = start
        interleaver.mediators = scheduled

    def unflatten(self, model: "VLLM") -> None:
        """Re-point each worker from its tokens to its row.

        Logits and sampled ids carry one row per *request*, not per token, so the
        spans that scoped the forward would select the wrong thing. The row order
        is the same order the forward's tensors used.
        """
        row = 0
        for _, workers in self._by_row():
            for mediator in workers:
                # Having a span is exactly what "scheduled" means: `scope` clears
                # every tracked worker's, `take` sets it for the ones it kept, and
                # nothing writes one between there and here.
                if mediator.batch_group is not None:
                    mediator.batch_group = [row, 1]
            row += 1

        model.interleaver.batcher.total = row

    @staticmethod
    def engine_key(worker_id: str, request_ids) -> Optional[tuple[str, int]]:
        """What the engine calls this request, and which of its sequences this is.

        The two sides name the same request differently, in two ways.

        vLLM appends a hash of the request's content on the way in, so a request
        the engine calls ``"0"`` arrives here as ``"0-a2460f0e"``. And ``n > 1``
        fans one request out into a child per sampled sequence, named
        ``"{index}_{parent}"`` (vLLM's ``ParentRequest``), while the engine only
        ever asks about — and only ever returns — the parent.

        The child reading is taken only when the parent it names is one the engine
        is actually asking about, so a request whose own id happens to start with
        digits and an underscore is not mistaken for somebody's second sequence.

        Returns:
            ``(engine_id, sequence_index)``, or None if this is not one of theirs.
        """
        engine_id = worker_id.rsplit("-", 1)[0]
        if engine_id in request_ids:
            return engine_id, 0
        index, _, parent = engine_id.partition("_")
        if parent and index.isdigit() and parent in request_ids:
            return parent, int(index)
        if worker_id in request_ids:
            return worker_id, 0
        return None

    def match(self, request_ids: set[str]) -> list[tuple[str, str, int]]:
        """Pair the engine's name for each of our requests with this worker's.

        Returns:
            ``(engine_id, worker_id, sequence_index)`` for each requested id that
            is ours. One engine id can appear several times — once per sampled
            sequence, when the request asked for more than one.
        """
        matched = []
        for worker_id in self.mediators:
            key = self.engine_key(worker_id, request_ids)
            if key is not None:
                matched.append((key[0], worker_id, key[1]))
        return matched

    def record_saves(self) -> None:
        """Note, on each scheduled worker, which of its values were saved.

        Read from the thread-local save-set while still on the thread the workers
        ran on. Collection can happen on another thread — Ray dispatches it through
        its own RPC worker — where that thread-local is empty, so the answer is
        captured here and carried on the mediator instead. The set only grows across
        a request's steps, so re-recording each step keeps the latest superset.
        """
        registered = [
            mediator
            for copies in self.registered.values()
            for mediator in copies.values()
        ]
        for mediator in list(self.mediators.values()) + registered:
            if mediator.batch_group is None:
                continue
            self.record(mediator)

    def record(self, mediator: Any) -> None:
        """Snapshot one worker's saved names, and its error if it has one.

        Split out because the snapshot is not only taken per step: a worker served
        at collect time — one parked on ``tracer.result``, which only exists once
        the engine has assembled the output — binds its name after the last
        `record_saves` of the run, and would otherwise be recorded as having
        saved nothing.
        """
        from ....intervention.errors import capture_exception

        saved = _saves()
        # Saves marked in this process, plus names the sending process
        # marked before serialization (Mediator.presaved).
        mediator.nnsight_saved = {
            name for name, value in mediator.lcls.items() if id(value) in saved
        } | mediator.presaved
        # An error (or stop) is captured on the workers' own thread too, for the
        # same reason saves are — the collect thread cannot read the exception's
        # intervention traceback off this greenlet. Captured once: an erred worker
        # stays scheduled (see Requests.scope) and would otherwise re-capture every
        # step until the request is retired.
        if (
            mediator.exception is not None
            and getattr(mediator, "nnsight_error", None) is None
        ):
            mediator.nnsight_error = capture_exception(mediator.exception)

    def harvest(self, finished: set[str]) -> None:
        """Move finished requests' registered values somewhere collectable.

        Driven by the scheduler's own finished set, and again by a collect that
        finds a request not yet harvested — the scheduler's pass happens at the
        top of the *next* step, which on the async path may come after the
        collect, or (for the last request in flight) never. Idempotent: a request
        already harvested is gone from `registered` and skipped.

        Off the scheduler rather than only off a collect so a registration works
        for requests nobody is tracing — the OpenAI server's,
        say — where nothing would otherwise come asking. A worker still parked
        when its request ends was waiting for a location this request never
        reached; that is ordinary for a registration (a block written for one
        architecture's layer, a request that stopped early), so it is unwound
        quietly and whatever it did save is kept.
        """
        for worker_id in list(self.registered):
            # Named either way: the scheduler says "0-a2460f0e", a collect asks
            # with the engine's "0".
            if self.engine_key(worker_id, finished) is None:
                continue
            for registration_id, mediator in self.registered.pop(worker_id).items():
                if registration_id not in self.harvested:
                    continue
                if mediator.alive:
                    self.finish_dangling_mediator(mediator)
                names = getattr(mediator, "nnsight_saved", set()) | mediator.presaved
                saved = {
                    name: mediator.lcls[name]
                    for name in names
                    if name in mediator.lcls
                }
                # A block that raised has to be reported. It saved nothing, so
                # without this the request would simply be missing from the
                # results and a broken registration would look like an idle one.
                error = getattr(mediator, "nnsight_error", None)
                if saved or error is not None:
                    self.harvested[registration_id][worker_id] = {
                        "saves": saved,
                        "error": error,
                    }

    def finish_dangling_mediator(self, mediator: Any) -> None:
        """Unwind a still-parked worker without surfacing an error to anyone.

        `finish_dangling` reports through the request its worker rode in on; a
        registered copy has no such request to report to, so it is only unwound
        (running its ``finally`` blocks) and its exception dropped.
        """
        from ....intervention.interleaver import OutOfOrderError

        try:
            mediator.worker.throw(
                OutOfOrderError("the request finished before this location was reached")
            )
        except BaseException:
            return

    def serve_result(self, worker_id: str, output: Any) -> None:
        """Hand a finished request's output to a worker parked on ``tracer.result``.

        A worker parked anywhere else is left alone — its own location is what it
        is waiting for, and `finish_dangling` reports it. Runs on the workers'
        own thread where the greenlet can be resumed; where that differs (Ray's
        collect) the switch is refused and the read is left unserved, exactly as
        it was before.
        """
        from greenlet import error as greenlet_error

        mediator = self.mediators.get(worker_id)
        if mediator is None or not mediator.alive:
            return
        pending = mediator.pending
        if pending is None or pending.provider != "result":
            return
        try:
            mediator.handle("result", output)
        except greenlet_error:
            return
        # The block ran on past its read and bound whatever it saved there, after
        # the run's last `record_saves` — so take the snapshot again or those
        # names go home missing.
        self.record(mediator)

    def finish_dangling(self, worker_id: str) -> None:
        """Surface a worker still parked when its request has finished.

        A worker still [`alive`][nnsight.intervention.interleaver.Mediator.alive] at the end was waiting on a location the model
        never reached — the interleaver's [`check_dangling_mediators`][nnsight.intervention.interleaver.Interleaver.check_dangling_mediators], but for a
        single request as it retires here rather than after a whole local run. Two
        cases, both unwound by throwing into the worker (so its ``finally`` blocks run):
        a plain read past the model's point is a real [`OutOfOrderError`][nnsight.intervention.interleaver.OutOfOrderError] kept as
        the request's deferred error so it reaches the client; a ``tracer.iter`` loop
        that outran generation is expected, so it only warns.

        Runs on the workers' own thread, where the greenlet can be resumed — the throw
        is skipped where that thread differs (e.g. Ray's collect), leaving the worker
        to be dropped without a surfaced error.
        """
        from greenlet import error as greenlet_error

        from ....intervention.errors import capture_exception
        from ....intervention.interleaver import Event, OutOfOrderError

        mediator = self.mediators.get(worker_id)
        if mediator is None or not mediator.alive:
            return

        # `requester` is printed into the error, where Pending renders as
        # "{location}.i{n}" — the occurrence is what explains an iter loop that
        # asked for more steps than the request generated.
        requester = mediator.pending
        if requester.event is Event.BARRIER:
            error: BaseException = ValueError(
                "A barrier was never reached by every block it waits for; "
                "check the count it was created with"
            )
            over_iterated = False
        else:
            error = OutOfOrderError(
                f"'{requester}' was requested but the model already ran past it"
            )
            over_iterated = mediator.iteration != 0

        try:
            mediator.worker.throw(error)
        except greenlet_error:
            return
        except BaseException as thrown:
            if over_iterated:
                warnings.warn(
                    f"'{requester}' was never reached: the model ran fewer iterations "
                    "than the loop requested. Values from reached iterations are kept."
                )
            else:
                mediator.nnsight_error = capture_exception(thrown)

    def saves(self, worker_id: str) -> dict:
        """This request's block-scope names that were marked with ``.save()``."""
        mediator = self.mediators.get(worker_id)
        if mediator is None:
            return {}
        saved = getattr(mediator, "nnsight_saved", set())
        return {name: mediator.lcls[name] for name in saved if name in mediator.lcls}

    def error(self, worker_id: str) -> Optional[dict]:
        """This request's deferred exception, captured for the client, or None."""
        mediator = self.mediators.get(worker_id)
        if mediator is None:
            return None
        return getattr(mediator, "nnsight_error", None)


class NNsightGPUModelRunner(GPUModelRunner):
    """A vLLM model runner that interleaves interventions with the forward."""

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        from vllm.tokenizers import cached_tokenizer_from_config

        from ..vllm import VLLM

        super().load_model(*args, **kwargs)

        # An Envoy tree over the real module. Passing a loaded module builds it
        # directly, so no weights are read twice and the paths match the ones the
        # client's meta tree gave the user. Building it here is also what walks
        # every module past `VLLMFragments.instrument`, so the tree comes back
        # already knowing which of its values are one rank's piece — on one rank
        # it finds nothing and stays inert.
        self.nnsight_model: VLLM = VLLM(
            self.model, interleaver=Interleaver(fragments=VLLMFragments())
        )
        self.nnsight_model.tokenizer = cached_tokenizer_from_config(self.model_config)

        interleaver = self.nnsight_model.interleaver
        # No envoy: the spans come from the scheduler rather than from an invoke,
        # so the row math never looks at a tree — and there is none to hand it yet.
        interleaver.batcher = VLLMBatcher(None)
        # A worker's error must end only its own request, not tear down the engine
        # every other request shares.
        interleaver.defer_exceptions = True

        self.nnsight_requests = Requests()
        # The map that resolves a serialized request's persistent ids (the interleaver,
        # every module, the tokenizer) back to this worker's objects. The tree is fixed
        # after load, so build it once here rather than walk it every step in `add`.
        self.nnsight_persistent_objects = (
            self.nnsight_model._remoteable_persistent_objects()
        )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        super()._update_states(scheduler_output)

        requests = self.nnsight_requests
        # Registered blocks are collected off the scheduler's own finished set:
        # a request nobody traced has no collect coming for it, and this is the
        # one place the runner is told it is over.
        finished = getattr(scheduler_output, "finished_req_ids", None)
        if finished and requests.registered:
            requests.harvest(set(finished))
        requests.add(
            scheduler_output.scheduled_new_reqs, self.nnsight_persistent_objects
        )
        # input_batch order, not the scheduler's: the batch is condensed and may be
        # reordered after the scheduler counts tokens, and the forward's tensors
        # follow the batch.
        requests.rows = list(self.input_batch.req_ids)
        requests.tokens = dict(scheduler_output.num_scheduled_tokens)
        requests.scope(self.nnsight_model)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional["IntermediateTensors"] = None,
    ) -> Any:
        # The worker runs each block directly as a mediator greenlet, with no Tracer
        # to open a trace scope, so a block's `.save()` would see no trace and raise.
        # Open one on this thread — the forward thread the greenlet is created and run
        # on, which need not be load_model's (under Ray it isn't). Idempotent: depth
        # persists per thread, and is left open (collect discards each request's saved
        # ids; the whole set is cleared below whenever nothing is in flight).
        if not getattr(_local, "depth", 0):
            inc()

        # `.save()` marks values by object id in a thread-local set that this thread
        # never clears via dec (depth stays open). collect discards the ids it returns,
        # but a bare `x.save()` or a loop-reassigned save marks a value that is never
        # collected, leaking its id — and a later request's value at a reused address
        # could then be mistaken for saved. Whenever the engine has drained to no
        # tracked requests, no pending save can matter, so clear the set outright; that
        # bounds its growth and stops any id reuse across separate waves of requests.
        # (Residual: reuse *within* one wave of concurrent requests, which is rare.)
        requests = self.nnsight_requests
        # Registered blocks count as in-flight too. They mark saves like any other
        # block, and a block that spans several steps (tracer.iter) is only marked
        # on the step it first bound the value — clearing underneath it would make
        # `record_saves` recompute an empty set and drop what it already had.
        if not requests.mediators and not requests.errored and not requests.registered:
            _saves().clear()

        interleaver = self.nnsight_model.interleaver
        # The scheduler picks this step's requests partway through the forward, so
        # there is nothing to register yet. Entering empty leaves the interleaver
        # with no worker to start — `Requests.scope` starts them as they appear.
        interleaver.mediators = []

        with interleaver:
            output = super().execute_model(scheduler_output, intermediate_tensors)
            # The forward is done; what follows it is per-request, not per-token.
            self.nnsight_requests.unflatten(self.nnsight_model)
        return output

    def sample_tokens(self, *args: Any, **kwargs: Any) -> Any:
        if self.execute_model_state is not None:
            original = self.execute_model_state.logits
            # Stays `original` if a tracer.stop() unwinds the handle before it
            # returns — the interleaver swallows the stop — so the step's own logits
            # are sampled unchanged.
            logits = original
            with self._still_running():
                # Serve this step's logits through the same `logits` eproperty the
                # client reads (VLLM.logits) — its `provide` hands the value to this
                # model's interleaver at the eproperty's own location, so the two
                # sides can't drift out of sync.
                model = self.nnsight_model
                logits = type(model).logits.provide(model, original)
            # The state is a namedtuple, so an edited tensor means a new one; an
            # untouched read hands the same tensor back and needs no rebuild.
            if logits is not original:
                state = self.execute_model_state
                self.execute_model_state = type(state)(
                    **{**state._asdict(), "logits": logits}
                )

        output = super().sample_tokens(*args, **kwargs)
        # Sampling closes the step: every block that was going to finish this step has,
        # whether it read activations, logits, or samples. Capture all their saves now,
        # in one pass, still on the workers' own thread (see Requests.record_saves).
        self.nnsight_requests.record_saves()
        return output

    def _sample(self, *args: Any, **kwargs: Any) -> Any:
        sampler_output = super()._sample(*args, **kwargs)

        with self._still_running():
            # Serve through the client's `samples` eproperty (see sample_tokens).
            model = self.nnsight_model
            sampler_output.sampled_token_ids = type(model).samples.provide(
                model, sampler_output.sampled_token_ids
            )

        self._finish_erred(sampler_output)
        return sampler_output

    def _finish_erred(self, sampler_output: Any) -> None:
        """End any request whose worker raised — a ``tracer.stop()`` or a real error.

        vLLM decides a request is done from the token it just sampled, so such a
        request is retired by forcing its sampled token to end-of-sequence: the
        scheduler's stop check then finishes it and schedules it no more. Whether the
        exception was an intentional stop or a real error only decides if it is
        re-raised at the client (see [`nnsight.intervention.errors`][nnsight.intervention.errors]). A worker that
        raised is no longer alive, so it is found on the tracked requests rather than
        among the still-running interleaver mediators; its row in this step's output is
        its ``batch_group`` (an erred worker is kept scheduled — see
        [`Requests.scope`][nnsight.modeling.vllm.model_runners.GPUModelRunner.Requests.scope] — so [`Requests.unflatten`][nnsight.modeling.vllm.model_runners.GPUModelRunner.Requests.unflatten] gives it one every step).

        The forced token is re-applied every step the request survives, because the
        scheduler's stop check can defer it: ``min_tokens`` holds the request open
        until that many tokens are produced, at which point the forced EOS stops it.
        ``ignore_eos`` is the one case EOS cannot end — such a request runs to
        ``max_tokens``, where it finishes naturally and its error is surfaced then;
        forcing a stop there would need vLLM's own abort, which does not surface a
        finished output on the synchronous engine. A tokenizer with no EOS token at all
        is the same story — nothing to force, so the request runs to ``max_tokens``.
        """
        eos = getattr(self.nnsight_model.tokenizer, "eos_token_id", None)
        if eos is None:
            return
        for mediator in self.nnsight_requests.mediators.values():
            if mediator.exception is not None and mediator.batch_group is not None:
                sampler_output.sampled_token_ids[mediator.batch_group[0]] = eos

    def _still_running(self) -> Any:
        """The interleaver, carrying only the workers still parked mid-block.

        The forward left a worker per scheduled request; for the per-request handles
        that follow (logits, samples) keep only the ones still parked, which may want
        those values. A finished block has nothing left to offer there. (A cache never
        needs them: ``Cache.observe`` records only module inputs/outputs.)
        """
        interleaver = self.nnsight_model.interleaver
        interleaver.mediators = [
            mediator for mediator in interleaver.mediators if mediator.alive
        ]
        return interleaver

    def collect_nnsight(
        self,
        request_ids: list[str],
        finished_request_ids: Optional[list[str]] = None,
        outputs: Optional[dict] = None,
    ) -> Optional[bytes]:
        """Return the saved values and any deferred error of the named requests.

        Keyed per request rather than merged, so two traces that happen to name a
        variable the same don't overwrite each other on the way home. Each entry is
        ``{"saves": {...}, "error": ..., "registered": {...}}``.

        A registered block's values are kept apart from the trace's own because
        they are not the same kind of thing: a name a *trace* saved on several
        requests is one shared container the invokes were writing into, and
        [`merge_shared_saves`][nnsight.modeling.vllm.collect.merge_shared_saves]
        reassembles it on that assumption. A registration saving the same name on
        every request it runs on looks identical from the outside and is not — it
        is one value per request — so merging them together would quietly fold a
        thousand separate activations into one.

        Args:
            request_ids: Requests to collect saved values from.
            finished_request_ids: Those that are done, whose workers are wound up
                and forgotten afterwards.
            outputs: Engine request id -> the ``RequestOutput`` that request
                produced, for serving ``tracer.result``. The engine has it and the
                worker does not, so it has to be handed back across; a caller that
                does not pass it simply leaves ``result`` unserved.
        """
        requests = self.nnsight_requests
        finished = set(finished_request_ids or [])
        wanted = set(request_ids) | finished
        collected: dict[str, dict] = {}

        def entry_for(request_id: str) -> dict:
            return collected.setdefault(
                request_id,
                {"saves": {}, "error": None, "registered": {}, "sequences": {}},
            )

        def sequence_of(entry: dict, index: int) -> dict:
            """One sampled sequence's values. Index 0 also fills the flat keys.

            ``saves``/``registered`` are the request's primary sequence, which is
            all there is unless it asked for several — so a caller that never uses
            ``n`` reads exactly what it always did.
            """
            sequence = entry["sequences"].setdefault(
                index, {"saves": {}, "registered": {}}
            )
            return entry if index == 0 else sequence

        # Harvest anything finished that the scheduler has not got to yet, so a
        # collect never reads an empty shelf for a request that is over.
        if finished and requests.registered:
            requests.harvest(finished)

        # Registered values, taken rather than read. This is the only way out for
        # them: the output they ride home on is the one the caller already has, so
        # nothing else will come asking and holding a second copy would just grow.
        # Answered from *every* rank — a registered block runs wherever the layers
        # it reads live, which under pipeline parallelism is not only rank 0 — and
        # the caller merges what the ranks return.
        for harvested in requests.harvested.values():
            for worker_id in list(harvested):
                key = requests.engine_key(worker_id, wanted)
                if key is None:
                    continue
                engine_id, index = key
                taken = harvested.pop(worker_id)
                entry = entry_for(engine_id)
                target = sequence_of(entry, index)
                target["registered"].update(taken["saves"])
                if index == 0:
                    entry["sequences"][0]["registered"].update(taken["saves"])
                if entry["error"] is None:
                    entry["error"] = taken["error"]

        # Everything below is the trace's own. Every rank ran the block and so
        # has a worker to wind up, but only one rank's values are wanted: the
        # reads are gathered, so every rank holds the same ones, and shipping
        # them all would cost bandwidth to hand back a copy that is thrown away.
        #
        # Only the *reporting* is gated. This used to be an early return, which
        # left every other rank's workers, their greenlets, and their captured
        # activations in place for the life of the engine — `get_pp_group().rank`
        # is the global rank when there is no pipeline, so under plain tensor
        # parallelism every rank but 0 skipped its own cleanup.
        reporting = get_pp_group().rank == 0

        matched = requests.match(wanted)

        # The run's return value, served the way [`Envoy.interleave`][nnsight.intervention.envoy.Envoy.interleave]
        # serves it locally — right after the model has produced it and before
        # anything is read back. It cannot be served there for this runtime: the
        # block runs here in the worker and the value is a `RequestOutput` the
        # *engine* assembles, so this is the first moment both exist. Served per
        # request, since each has its own.
        for engine_id, worker_id, index in matched:
            if engine_id in finished and outputs and engine_id in outputs:
                output = outputs[engine_id]
                # What a registered block saved for this request, put on the copy
                # the block is about to be handed. It is the one thing a trace
                # cannot reach any other way: the engine attaches it to *its* copy
                # after this returns, and the trace never sees that object.
                #
                # The trace's own saves are not here and cannot be: they are what
                # this call is collecting, and one of them may be the output
                # itself.
                # This sequence's own, when the request asked for several: each
                # child ran its own copy of the block and is owed its own values.
                entry = collected.get(engine_id) or {}
                sequence = (entry.get("sequences") or {}).get(index) or entry
                registered = sequence.get("registered")
                if registered:
                    output.saves = dict(registered)
                requests.serve_result(worker_id, output)

        # A worker still parked when its request finishes was waiting on a location the
        # model never reached; surface that as its deferred error before it is read.
        for engine_id, worker_id, _ in matched:
            if engine_id in finished:
                requests.finish_dangling(worker_id)

        saved = _saves()
        for engine_id, worker_id, index in matched:
            values = requests.saves(worker_id)
            if reporting:
                entry = entry_for(engine_id)
                sequence_of(entry, index)["saves"] = values
                entry["sequences"][index]["saves"] = values
                if entry["error"] is None:
                    entry["error"] = requests.error(worker_id)
            if engine_id in finished:
                # Drop this request's saved values from the thread-local set as they
                # leave: it is keyed by object id, so a finished request's ids left
                # behind could be reused by a later request's values and mistaken for
                # saved. (No-op on a collect thread other than the workers' own, e.g.
                # Ray, where that set is empty — but harmless.)
                for value in values.values():
                    saved.discard(id(value))
                requests.mediators.pop(worker_id, None)

        # Requests whose payload failed to deserialize (see Requests.add) carry no
        # worker; surface their captured error here, keyed like the collected saves.
        for req_id, error in list(requests.errored.items()):
            key = Requests.engine_key(req_id, wanted)
            if key is None:
                continue
            if reporting:
                entry_for(key)["error"] = error
            if Requests.engine_key(req_id, finished) is not None:
                requests.errored.pop(req_id, None)

        # Saves may still be device work in flight; they are about to be pickled.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return pickle.dumps(collected)

    # ------------------------------------------------------------------
    # Worker-side RPC entry points (called by name via collective_rpc)
    # ------------------------------------------------------------------

    def nnsight_register(self, registration_id: str, payload: bytes) -> None:
        """Keep a block and run it for every request from now on.

        See [`nnsight.modeling.vllm.registration`][nnsight.modeling.vllm.registration].
        Runs on all ranks, so every rank builds the same per-request copies and
        their reads stay in lockstep — which is what a sharded model's gathers
        need.
        """
        self.nnsight_requests.register(
            registration_id, payload, self.nnsight_persistent_objects
        )

    def nnsight_clear_registered(self, registration_id: str) -> None:
        """Stop running a block and drop what it has not handed back."""
        self.nnsight_requests.unregister(registration_id)

    def nnsight_request_count(self) -> int:
        """How many requests' workers this runner still tracks.

        A leak gauge: it should return to zero once every request has finished or
        been aborted. A number that only grows across requests means workers are
        outliving their requests — a finished one is freed in `collect_nnsight`,
        an aborted one when its stream is closed (see the async and serve backends).
        """
        return len(self.nnsight_requests.mediators)

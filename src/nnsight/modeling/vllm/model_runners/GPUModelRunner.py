"""Run interventions inside vLLM's worker, against the real weights.

This is where a trace written in another process actually happens. The runner
builds its own [`VLLM`][nnsight.modeling.vllm.vllm.VLLM] over the module vLLM
loaded, so the module tree here has the same paths the client wrote against; a
worker arriving on a request then resolves straight onto the real modules.

Three points on vLLM's own path carry it:

* ``_update_states`` — the scheduler has just decided what runs this step, so new
  requests hand over their workers and every worker's token span is recomputed.
* ``execute_model`` — the forward, run with the interleaver open so hooks serve
  the parked workers; ``_model_forward`` re-points them from tokens to rows as
  soon as the forward returns, before the logits are computed.
* ``_sample`` — logits and sampled ids never pass through a module, so they are
  offered to workers directly by location.

Workers run as greenlets on this thread. ``_update_states`` is called from
``execute_model``, and hooks fire on whichever thread runs the forward, so the
worker and the model take strict turns on one thread — there is nothing to
synchronize *during the forward*. Collection (``collect_nnsight``) runs on that
thread too, between steps, which is what lets it resume a worker to serve
``tracer.result`` or unwind it; the one executor where it would not, Ray's
compiled-graph one, is refused (see ``NNsightGPUWorker``).
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any

import torch
from vllm import envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from ....intervention.errors import capture_exception
from ....intervention.interleaver import Mediator, OutOfOrderError, dangling_unwind
from ....intervention.serialization import loads
from ....tracing.tracer import _local, inc
from ..batching import VLLMBatcher
from ..collect import dumps_compact
from ..fragments import VLLMFragments
from ..interleaver import VLLMInterleaver
from ..vllm import VLLM

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput


class Request:
    """What this engine carries for one of its requests.

    vLLM names a request ``"{external}-{8 hex}"`` on the way in and an ``n > 1``
    child ``"{index}_{parent}"``, while the engine asks about — and returns —
    the external id; both spellings are parsed here once. ``mediator`` is the
    traced worker, if the request is an nnsight trace; ``copies`` are the
    registered blocks' per-request workers, moved to ``harvested`` once the
    request is over; ``error`` is a payload that failed to deserialize.
    """

    __slots__ = ("id", "stem", "engine_id", "index", "mediator", "copies", "harvested", "error")

    def __init__(self, request_id: str) -> None:
        self.id = request_id
        # vLLM's suffix is eight hex digits after a dash, and is not added at all
        # under VLLM_DISABLE_REQUEST_ID_RANDOMIZATION; anything else after a dash
        # is the caller's own id.
        suffix = (
            None
            if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION
            else re.search(r"-[0-9a-f]{8}$", request_id)
        )
        self.stem = request_id[: suffix.start()] if suffix else request_id
        index, _, parent = self.stem.partition("_")
        self.engine_id, self.index = (parent, int(index)) if parent and index.isdigit() else (self.stem, 0)
        self.mediator: Any = None
        self.copies: dict[str, Any] = {}
        self.harvested: dict[str, dict] = {}
        self.error: dict | None = None

    def named(self, ids) -> bool:
        """Whether ``ids`` (the engine's, or the scheduler's) name this request."""
        return self.engine_id in ids or self.stem in ids or self.id in ids

    def key(self, ids) -> tuple[str, int]:
        """``(engine_id, sequence index)`` as the engine that asked about ``ids`` knows it.

        The child reading is taken only when the parent it names is one the engine
        asked about, so an id that merely starts with digits and an underscore is
        not mistaken for somebody's second sequence.
        """
        return (self.engine_id, self.index) if self.engine_id in ids else (self.stem, 0)

    def workers(self) -> list:
        """Registered copies first, so a trace reads what they left behind."""
        return list(self.copies.values()) + ([self.mediator] if self.mediator is not None else [])

    def saves(self) -> dict:
        """The trace's block-scope values that were marked with ``.save()``."""
        return {} if self.mediator is None else _saved(self.mediator)

    def deferred(self) -> dict | None:
        """The request's deferred error, captured for the client, or None.

        The request's own error first — a block that failed to deserialize, or an
        ``edits=`` name nothing is installed under — then the traced block's.
        """
        if self.error is not None:
            return self.error
        return None if self.mediator is None else _error(self.mediator)


def _saved(mediator: Any) -> dict:
    """A worker's block-scope values marked with ``.save()``, here or before it was sent (``presaved``)."""
    marked = mediator.saves or ()
    return {
        name: value
        for name, value in mediator.lcls.items()
        if id(value) in marked or name in mediator.presaved
    }


def _error(mediator: Any) -> dict | None:
    """A worker's exception, captured for the client, or None."""
    return None if mediator.exception is None else capture_exception(mediator.exception)


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
        requests: Request id -> `Request`, for as long as anything of it is
            still wanted.
        templates: Registration id -> the deserialized block each request's copy
            is built from, so the source is compiled once rather than per request.
        names: Registration id -> the name it was installed under, or ``None``.
            A request that names the edits it wants (``extra_args["nnsight_edits"]``)
            gets copies of those and of every unnamed one; a request that names
            none gets copies of them all.
    """

    def __init__(self) -> None:
        self.requests: dict[str, Request] = {}
        self.templates: dict[str, Any] = {}
        self.names: dict[str, str | None] = {}
        # Workers whose request is out of the batch (preempted), and the
        # interleaver's counts when that was last noted, so the visits they sit
        # out are subtracted from their own count (see `scope`).
        self.out: set = set()
        self.counts: dict[str, int] = {}
        # Rows in this step's batch, for `unflatten`.
        self.nrows = 0

    def register(
        self,
        registration_id: str,
        payload: bytes,
        persistent_objects: dict,
        name: str | None = None,
    ) -> None:
        """Take a block the engine should run for every request from now on."""
        self.templates[registration_id] = loads(payload, persistent_objects=persistent_objects)
        self.names[registration_id] = name

    def unregister(self, registration_id: str) -> None:
        """Stop running a block and forget anything it has not handed back."""
        self.templates.pop(registration_id, None)
        self.names.pop(registration_id, None)
        for request in self.requests.values():
            # A copy may be parked mid-request; dropped without an unwind, it stays.
            self.finish_dangling(request.copies.pop(registration_id, None), quiet=True)
            request.harvested.pop(registration_id, None)

    def add(self, new_requests: list["NewRequestData"], persistent_objects: dict) -> None:
        """Take the worker off each new request that carries one.

        A request with no nnsight payload is another tenant of the same engine and
        runs only the registered blocks — it still occupies tokens in the batch,
        so [`scope`][nnsight.modeling.vllm.model_runners.GPUModelRunner.Requests.scope]
        counts it. A payload that fails to deserialize is recorded as that
        request's error and surfaced at collect, rather than raised inside
        ``execute_model`` where it would take the engine every tenant shares down.

        A request this engine already carries is one vLLM preempted and is
        resuming; its workers continue (the engine replays the tokens they already
        saw inside one recompute step, so a fresh block would be short by exactly
        those steps — `scope` keeps the steps they sat out off their count).
        """
        for data in new_requests:
            if data.req_id in self.requests:
                continue
            request = self.requests[data.req_id] = Request(data.req_id)
            extra_args = getattr(data.sampling_params, "extra_args", None) or {}
            # Which installed blocks this request runs: all of them, unless it
            # named the ones it wants — then those, plus every block installed
            # without a name. A name nothing is installed under is the request's
            # error rather than a silent no-op, surfaced at collect like any other.
            wanted = extra_args.get("nnsight_edits")
            if wanted is not None:
                installed = {name for name in self.names.values() if name is not None}
                unknown = [name for name in wanted if name not in installed]
                if unknown:
                    request.error = capture_exception(
                        ValueError(
                            f"edits={list(wanted)!r} names {unknown!r}, but no edit "
                            f"is installed under that name (installed: "
                            f"{sorted(installed)!r}). Install it with "
                            "model.edit(name=...), or drop it from the list."
                        )
                    )
            for registration_id, template in self.templates.items():
                name = self.names.get(registration_id)
                if wanted is not None and name is not None and name not in wanted:
                    continue
                # A scope of its own, down to the containers bound outside the
                # block: otherwise every request appends to one list, and each
                # output carries every earlier request's values too.
                scope = {
                    name: value.copy() if isinstance(value, (list, dict, set)) else value
                    for name, value in template.lcls.items()
                }
                copy = Mediator(template.code, template.glbls, scope)
                copy.presaved = set(template.presaved)
                request.copies[registration_id] = copy
            if "nnsight_mediator" not in extra_args:
                continue
            try:
                request.mediator = loads(extra_args["nnsight_mediator"], persistent_objects=persistent_objects)
            except Exception as exception:
                request.error = capture_exception(exception)

    def refuse_chunked(self, spans: list[tuple[str, int]], states: dict) -> None:
        """Give a request whose prompt this step only partly prefills its error, not a worker.

        Chunked prefill (off by default on an nnsight engine) splits a prompt
        across steps: a block would see a slice of its prompt now and the rest
        later, and the sample of every chunk but the last is one vLLM discards.
        ``states`` is the runner's per-request state, which knows how much of the
        prompt is computed before this step.
        """
        for request_id, scheduled in spans:
            request = self.requests.get(request_id)
            state = states.get(request_id)
            if request is None or state is None or request.mediator is None:
                continue
            if state.num_computed_tokens + scheduled >= state.num_prompt_tokens:
                continue
            request.error = capture_exception(
                RuntimeError(
                    f"This request's {state.num_prompt_tokens}-token prompt was split "
                    f"across steps by chunked prefill ({scheduled} tokens this step), "
                    "so no block could see it whole. nnsight engines disable chunked "
                    "prefill by default; drop enable_chunked_prefill=True, or raise "
                    "max_num_batched_tokens past the prompt length."
                )
            )
            request.mediator = None
            request.copies = {}

    def workers(self) -> list:
        """Every worker this engine is carrying, traced and registered alike."""
        return [mediator for request in self.requests.values() for mediator in request.workers()]

    def scope(self, model: "VLLM", spans: list[tuple[str, int]]) -> None:
        """Point every worker at its own tokens within this step's batch.

        ``spans`` is each scheduled request and its token count, in the order the
        forward's tensors follow. A worker's span is only meaningful for the step
        it was computed in, so every worker's is recomputed: workers whose request
        isn't running now report no group and are dropped from the interleaver, and
        a block that already ran to completion is dropped too, since the
        interleaver starts anything not alive and a finished block must not run a
        second time.
        """
        workers = self.workers()
        for mediator in workers:
            mediator.batch_group = None

        interleaver = model.interleaver
        scheduled = []
        start = 0

        def take(mediator: Any, row: int, tokens: int) -> None:
            """Give one worker this request's span, starting it the first time."""
            # A finished block is dropped — its work is done and it must not run
            # again — unless it holds open caches (tracer.cache() observes every
            # step until the request ends) or raised: an erred worker keeps a row
            # so _finish_erred can force its end-of-sequence every step until vLLM
            # actually retires it (once is not enough when min_tokens defers the
            # stop). `worker is not None` means started; started-but-not-alive
            # means finished.
            started = mediator.worker is not None
            finished = started and not mediator.alive
            if finished and not mediator.caches and mediator.exception is None:
                return
            mediator.batch_group = [start, tokens]
            mediator.row = row
            scheduled.append(mediator)
            # Which requests run is only settled here, once the forward has
            # already begun, so a worker is started the moment its request is
            # first scheduled rather than on the way into the interleaver.
            if not started:
                # Its saves are marked in a set of its own: the thread's is shared
                # by every request on the engine and never cleared by a trace exit.
                mediator.saves = set()
                try:
                    mediator.start(interleaver)
                except Exception as exception:
                    # A block that errors before it first parks (a bad line at
                    # the top) is deferred like one that errors mid-run; the
                    # runner's _finish_erred ends it.
                    if not interleaver.defer_exceptions:
                        raise
                    mediator.exception = exception

        for row, (request_id, tokens) in enumerate(spans):
            request = self.requests.get(request_id)
            if request is not None:
                for mediator in request.workers():
                    take(mediator, row, tokens)
            start += tokens

        # A started worker whose request is out of this step (vLLM preempted it)
        # must not count the step's visits as its own: move its `base` past
        # whatever passed since the last time this was noted, then note it again
        # for whoever is out now.
        counts = interleaver.counts
        for mediator in self.out:
            for location, count in counts.items():
                mediator.counts_at_start[location] = mediator.counts_at_start.get(location, 0) + count - self.counts.get(location, 0)
        self.out = {m for m in workers if m.worker is not None and m.batch_group is None and m.alive}
        if self.out:
            self.counts = dict(counts)

        # Every scheduled request's tokens, nnsight's or not — the leading dim of
        # the activations a worker will be narrowed out of.
        interleaver.batcher.total = start
        interleaver.mediators = scheduled
        interleaver.reindex()  # swapped mid-run: the indexes and `busy` follow the new list
        self.nrows = len(spans)

    def unflatten(self, model: "VLLM") -> None:
        """Re-point each worker from its tokens to its row.

        Logits and sampled ids carry one row per *request*, not per token, so the
        spans that scoped the forward would select the wrong thing.
        """
        for mediator in model.interleaver.mediators:
            mediator.batch_group = [mediator.row, 1]
        model.interleaver.batcher.total = self.nrows

    def harvest(self, finished: set[str]) -> None:
        """Shelve finished requests' registered values until they are collected.

        Driven by the scheduler's own finished set, and again by a collect that
        finds a request not yet harvested — the scheduler's pass happens at the
        top of the *next* step, which on the async path may come after the
        collect, or (for the last request in flight) never. Off the scheduler
        rather than only off a collect so a registration works for requests
        nobody is tracing — the OpenAI server's, say — where nothing would
        otherwise come asking. A copy still parked when its request ends was
        waiting for a location this request never reached, which is ordinary for
        a registration, so it is unwound quietly and whatever it saved is kept.
        """
        for request in self.requests.values():
            if not request.copies or not request.named(finished):
                continue
            for registration_id, mediator in request.copies.items():
                self.finish_dangling(mediator, quiet=True)
                saved = _saved(mediator)
                # A block that raised has to be reported: it saved nothing, and
                # without this a broken registration would look like an idle one.
                error = _error(mediator)
                if saved or error is not None:
                    request.harvested[registration_id] = {"saves": saved, "error": error}
            request.copies = {}
        # A request nobody traced that left nothing behind — another tenant's,
        # say — is over and owed nothing, and no collect will come asking about
        # it, so it is dropped here, off the scheduler.
        for request in list(self.requests.values()):
            if request.named(finished) and request.mediator is None and request.error is None and not request.harvested:
                del self.requests[request.id]

    def serve_result(self, mediator: Any, output: Any) -> None:
        """Hand a finished request's output to a worker parked on ``tracer.result``.

        A worker parked anywhere else is left alone — its own location is what it
        is waiting for, and `finish_dangling` reports it.
        """
        if mediator is None or not mediator.alive:
            return
        pending = mediator.pending
        if pending is None or pending.provider != "result":
            return
        try:
            mediator.handle("result", output)
        except Exception as exception:
            # Raised by the block after its read: this request's error, deferred
            # like any other, not the collect's, which every request in it shares.
            mediator.exception = exception

    def finish_dangling(self, mediator: Any, taps: frozenset = frozenset(), quiet: bool = False) -> None:
        """Surface a worker still parked when its request has finished.

        A worker still [`alive`][nnsight.intervention.interleaver.Mediator.alive] at the end was waiting on a location the model
        never reached — the interleaver's [`check_dangling_mediators`][nnsight.intervention.interleaver.Interleaver.check_dangling_mediators], but for a
        single request as it retires here rather than after a whole local run. Which
        of those a parked worker is, and what it should be told, is
        [`dangling_unwind`][nnsight.intervention.interleaver.dangling_unwind]'s to
        decide, so a trace behaves the same on this engine as it does locally: a read
        past the model's point is an error, and a ``tracer.iter`` loop the request
        could not supply — bounded and open alike — is cut short with a warning,
        keeping the values from the steps that ran. Surfacing differs, since the
        client is in another process: an error becomes the request's deferred error
        and is raised there.

        ``taps`` is the engine's tap set when it replays CUDA graphs: a module
        location outside it is never reached by a replayed step, and the error
        says so rather than reporting the model ran past it. ``quiet`` only
        unwinds (a registered copy has no request to report to).
        """
        if mediator is None or not mediator.alive:
            return

        # A worker whose request is over but which is parked nowhere already erred:
        # the interleaver deferred its exception and cleared its pending, and that
        # error — not a dangling read — is what its request goes home with. Still
        # alive means the error was raised outside the block (a swap that did not
        # fit its rows), so the block is still suspended; end it, or its greenlet
        # and everything its scope holds are never freed.
        requester = mediator.pending
        if requester is None:
            try:
                mediator.worker.throw()
            except BaseException:
                pass
            return

        error, expected = dangling_unwind(mediator)
        if (
            taps
            # A barrier unwinds with a ValueError and names no module location.
            and isinstance(error, OutOfOrderError)
            and requester.provider not in taps
            and requester.provider.rsplit(".", 1)[-1] in ("input", "output", "skip")
        ):
            # Outside the taps nothing is reached however many steps ran, so the
            # loop's shape is beside the point: this is the whole reason.
            error, expected = (
                OutOfOrderError(
                    f"'{requester.provider}' is not a tap on this engine, so a replayed "
                    "CUDA graph never reaches it. Declare it at construction — "
                    "VLLM(..., taps=[...]) — or trace an engine built with "
                    "enforce_eager=True, which serves every location."
                ),
                None,
            )

        try:
            mediator.worker.throw(error)
        except BaseException as thrown:
            if quiet:
                return
            # `thrown is error` only when the block let the unwind through; anything
            # a `finally` raised on the way out is the request's error instead.
            if expected is not None and thrown is error:
                warnings.warn(expected)
            else:
                mediator.exception = thrown



class NNsightGPUModelRunner(GPUModelRunner):
    """A vLLM model runner that interleaves interventions with the forward."""

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)

        # An Envoy tree over the real module. Passing a loaded module builds it
        # directly, so no weights are read twice and the paths match the ones the
        # client's meta tree gave the user. Building it here is also what walks
        # every module past `VLLMFragments.instrument`, so the tree comes back
        # already knowing which of its values are one rank's piece — on one rank
        # it finds nothing and stays inert.
        # The taps the client declared, if the engine replays CUDA graphs; they
        # ride the engine config so every worker process sees the same set.
        taps = self.vllm_config.additional_config.get("nnsight_taps", ())
        # `get_model`, not `self.model`: under CUDA graphs vLLM has wrapped the
        # module in its graph runner, and the tree is built over the module.
        self.nnsight_model: VLLM = VLLM(
            self.get_model(),
            interleaver=VLLMInterleaver(taps, fragments=VLLMFragments()),
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

    def capture_model(self) -> Any:
        """Record vLLM's CUDA graphs with the interleaver open.

        A module hands off only while interleaving, and recording the graphs runs
        the forward outside any step — so open the interleaver (with no workers)
        for the recording, which is what lets `VLLMInterleaver.handle` register
        each tap's replay into the graph.
        """
        interleaver = self.nnsight_model.interleaver
        with interleaver:
            # No workers, but the handoff must run: it is what registers the taps.
            interleaver.busy = True
            return super().capture_model()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> Any:
        # vLLM returns a correction to run once the forward is launched (speculative
        # decoding's token counts); `execute_model` calls it, so it goes back as is.
        corrections = super()._update_states(scheduler_output)

        requests = self.nnsight_requests
        # Registered blocks are collected off the scheduler's own finished set:
        # a request nobody traced has no collect coming for it, and this is the
        # one place the runner is told it is over.
        if scheduler_output.finished_req_ids:
            requests.harvest(scheduler_output.finished_req_ids)
        requests.add(
            scheduler_output.scheduled_new_reqs, self.nnsight_persistent_objects
        )
        # input_batch order, not the scheduler's: the batch is condensed and may be
        # reordered after the scheduler counts tokens, and the forward's tensors
        # follow the batch.
        tokens = scheduler_output.num_scheduled_tokens
        spans = [(rid, tokens[rid]) for rid in self.input_batch.req_ids]
        requests.refuse_chunked(spans, self.requests)
        requests.scope(self.nnsight_model, spans)
        return corrections

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> Any:
        # The worker runs each block directly as a mediator greenlet, with no Tracer
        # to open a trace scope, so a block's `.save()` would see no trace and raise.
        # Open one on this thread — the forward thread the greenlet is created and run
        # on, which need not be load_model's (under Ray it isn't). Idempotent: depth
        # persists per thread, and is left open; each worker marks its saves in a set
        # of its own (see `Requests.scope`), so none collect in the thread's.
        if not getattr(_local, "depth", 0):
            inc()

        interleaver = self.nnsight_model.interleaver
        # The scheduler picks this step's requests partway through the forward, so
        # there is nothing to register yet. Entering empty leaves the interleaver
        # with no worker to start — `Requests.scope` starts them as they appear.
        interleaver.mediators = []

        with interleaver:
            return super().execute_model(scheduler_output, intermediate_tensors)

    def _model_forward(self, *args: Any, **kwargs: Any) -> Any:
        output = super()._model_forward(*args, **kwargs)
        # The forward is done; what follows it — `compute_logits`, which runs the
        # model's `logits_processor` on one row per request, then the sampler —
        # is per-request, not per-token.
        self.nnsight_requests.unflatten(self.nnsight_model)
        return output

    def _sample(self, logits: Any, *args: Any, **kwargs: Any) -> Any:
        model = self.nnsight_model
        interleaver = model.interleaver
        # Re-entered for the per-request values; a worker that finished in the
        # forward is parked nowhere and so is not served. Both are served through
        # the eproperties the client reads (VLLM.logits, VLLM.samples), whose
        # `provide` hands the value over at the eproperty's own location, so the
        # two sides can't drift out of sync. A tracer.stop() that unwinds a handle
        # is swallowed by the interleaver, and the value stays as it was. vLLM has
        # applied a structured-output request's grammar mask by now, so a block
        # reads, and edits, the masked logits.
        with interleaver:
            logits = type(model).logits.provide(model, logits)
        # The sampler scales its logits in place (temperature, top-k, top-p), so
        # with a block in the step it gets a copy: a block's read is a view of
        # these, and would otherwise be scaled after the block saved it.
        if interleaver.mediators:
            logits = logits.clone()
        sampler_output = super()._sample(logits, *args, **kwargs)

        with interleaver:
            sampler_output.sampled_token_ids = type(model).samples.provide(
                model, sampler_output.sampled_token_ids
            )

        self._finish_erred(sampler_output)
        return sampler_output

    def _finish_erred(self, sampler_output: Any) -> None:
        """End any request whose worker raised — a ``tracer.stop()`` or a real error.

        vLLM decides a request is done from the token it just sampled, so such a
        request is retired by forcing its sampled token to the end-of-sequence id the
        scheduler's stop check compares against — the request's own
        ``sampling_params.eos_token_id`` — which then finishes it. Whether the
        exception was an intentional stop or a real error only decides if it is
        re-raised at the client (see [`nnsight.intervention.errors`][nnsight.intervention.errors]). Only a
        request's traced worker ends it — a registered copy's error stays its own — so
        they are found on the tracked requests; its row in this step's output is
        its ``batch_group`` (an erred worker is kept scheduled — see
        [`Requests.scope`][nnsight.modeling.vllm.model_runners.GPUModelRunner.Requests.scope] — so [`Requests.unflatten`][nnsight.modeling.vllm.model_runners.GPUModelRunner.Requests.unflatten] gives it one every step).

        The forced token is re-applied every step the request survives, because the
        scheduler's stop check can defer it: ``min_tokens`` holds the request open
        until that many tokens are produced, at which point the forced EOS stops it.
        A request with no eos id — ``ignore_eos`` clears it — has nothing to force: it
        runs to ``max_tokens``, where it finishes naturally and its error is surfaced
        then; forcing a stop there would need vLLM's own abort, which does not surface
        a finished output on the synchronous engine.
        """
        for request in self.nnsight_requests.requests.values():
            mediator = request.mediator
            if mediator is None or mediator.exception is None or mediator.batch_group is None:
                continue
            eos = self.requests[request.id].sampling_params.eos_token_id
            if eos is not None:
                sampler_output.sampled_token_ids[mediator.batch_group[0]] = eos

    def collect_nnsight(
        self,
        request_ids: list[str],
        finished_request_ids: list[str] | None = None,
        outputs: dict | None = None,
    ) -> bytes | None:
        """Return the saved values and any deferred error of the named requests.

        Keyed per request rather than merged, so two traces that happen to name a
        variable the same don't overwrite each other on the way home. Each entry is
        ``{"error": ..., "sequences": {index: {"saves": {...}, "registered": {...}}}}``,
        one sequence per sampled ``n`` (just ``0`` unless the request asked for
        several), plus flat ``saves`` and ``registered``: the primary sequence's.

        Every rank ran the block and so has workers to wind up, which is why every
        rank is called; an early return on the others once left their workers in
        place for the life of the engine. But only the first tensor-parallel rank
        replies with values, and the rest return ``None``: every rank runs every
        layer and its reads are gathered across the group, so each holds the same
        values, and pickling them on every rank would only be thrown away.

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
            return collected.setdefault(request_id, {"error": None, "sequences": {}})

        def note(entry: dict, error: dict | None) -> None:
            """Keep the request's first error, but a real one over a stop, which the client does not raise."""
            if error is not None and (entry["error"] is None or entry["error"]["is_control_flow"]):
                entry["error"] = error

        def sequence_of(entry: dict, index: int) -> dict:
            """One sampled sequence's values."""
            return entry["sequences"].setdefault(index, {"saves": {}, "registered": {}})

        # Harvest anything finished that the scheduler has not got to yet, so a
        # collect never reads an empty shelf for a request that is over.
        if finished:
            requests.harvest(finished)

        taps = self.nnsight_model.interleaver.taps
        for request in list(requests.requests.values()):
            if not request.named(wanted):
                continue
            done = request.named(finished)
            engine_id, index = request.key(wanted)
            entry = entry_for(engine_id)
            sequence = sequence_of(entry, index)
            # Registered values, taken rather than read: the output they ride
            # home on is the one the caller already has, so nothing else will
            # come asking and holding a second copy would just grow.
            for taken in request.harvested.values():
                sequence["registered"].update(taken["saves"])
                note(entry, taken["error"])
            request.harvested = {}

            mediator = request.mediator
            if mediator is not None:
                if done and outputs and engine_id in outputs:
                    # The run's return value, served the way `Envoy.interleave`
                    # serves it locally — the block runs here in the worker and
                    # the value is a `RequestOutput` the *engine* assembles, so
                    # this is the first moment both exist. What a registered
                    # block saved for this request goes on the copy the block is
                    # about to be handed; the engine attaches it to *its* copy
                    # after this returns and the trace never sees that object.
                    output = outputs[engine_id]
                    if sequence["registered"]:
                        output.saves = dict(sequence["registered"])
                    requests.serve_result(mediator, output)
                if done:
                    # Still parked when its request finished: waiting on a
                    # location the model never reached — its deferred error.
                    requests.finish_dangling(mediator, taps)
                sequence["saves"] = request.saves()
            note(entry, request.deferred())
            if done and not request.copies:
                requests.requests.pop(request.id, None)
        # A worker that sat out the step before its request ended is still in `out`,
        # holding its scope, until a next step that an idle engine never takes.
        requests.out = {mediator for mediator in requests.out if mediator.alive}

        if not get_tp_group().is_first_rank:
            return None
        # The flat keys are the primary sequence, which is all there is unless the
        # request asked for several; `attach` reads those. A request
        # can lack sequence 0: under `n > 1`, a child nothing traced whose
        # registered copies saved nothing is dropped at harvest.
        for entry in collected.values():
            primary = entry["sequences"].get(0, {"saves": {}, "registered": {}})
            entry["saves"], entry["registered"] = primary["saves"], primary["registered"]

        # Saves may still be device work in flight; they are about to be pickled.
        torch.cuda.synchronize()

        try:
            return dumps_compact(collected)
        except Exception:
            # A value that will not pickle is its own request's error. Raised from
            # here it would fail the whole collect, and at tp > 1 leave the other
            # ranks' replies queued for the next call to read as its own.
            for request_id, entry in collected.items():
                try:
                    dumps_compact(entry)
                except Exception as exception:
                    collected[request_id] = {
                        "error": capture_exception(exception),
                        "sequences": {}, "saves": {}, "registered": {},
                    }
            return dumps_compact(collected)

    # ------------------------------------------------------------------
    # Worker-side RPC entry points (called by name via collective_rpc)
    # ------------------------------------------------------------------

    def nnsight_register(
        self, registration_id: str, payload: bytes, name: str | None = None
    ) -> None:
        """Keep a block and run it for every request from now on.

        See [`nnsight.modeling.vllm.registration`][nnsight.modeling.vllm.registration].
        Runs on all ranks, so every rank builds the same per-request copies and
        their reads stay in lockstep — which is what a sharded model's gathers
        need. ``name`` is what requests may address it by (``edits=[...]``).
        """
        self.nnsight_requests.register(
            registration_id, payload, self.nnsight_persistent_objects, name=name
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
        return len(self.nnsight_requests.requests)

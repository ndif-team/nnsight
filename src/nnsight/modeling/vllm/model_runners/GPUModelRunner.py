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
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from ....intervention.serialization import loads
from ....tracing.tracer import _local, _saves, inc
from ..batching import VLLMBatcher

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

    from ..vllm import VLLM


def _parallel_linears(module: torch.nn.Module) -> list[torch.nn.Module]:
    """Every tensor-parallel linear in ``module``'s tree."""
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    return [
        child
        for child in module.modules()
        if isinstance(child, (ColumnParallelLinear, RowParallelLinear))
    ]


def _watch_parallel_linears(module: torch.nn.Module, batcher: VLLMBatcher) -> None:
    """Have the batcher note each parallel linear's value *before* workers see it.

    Registered before the interleaver's hooks, so — pre-forward and forward hooks
    both firing in registration order — these run first: the value is gathered by
    the time a worker reads it. There is no ordered-hook machinery to arrange this;
    it falls out of building the Envoy tree (which registers the interleaver's
    hooks) after this call and before `_release_parallel_linears`.
    """
    for linear in _parallel_linears(module):
        linear.register_forward_pre_hook(
            lambda mod, args, kwargs, batcher=batcher: batcher.watch(mod, "input"),
            with_kwargs=True,
        )
        linear.register_forward_hook(
            lambda mod, args, kwargs, out, batcher=batcher: batcher.watch(mod, "output"),
            with_kwargs=True,
        )


def _release_parallel_linears(module: torch.nn.Module, batcher: VLLMBatcher) -> None:
    """Have the batcher re-shard each parallel linear's value *after* workers edit it.

    Registered after the interleaver's hooks so they run last, handing vLLM back a
    shard of whatever the workers left — the reduced form its own forward expects.
    """
    for linear in _parallel_linears(module):
        # The interleaver serves a module's input as the whole (args, kwargs), so a
        # pre-hook returning that pair replaces both — hand back the resharded pair.
        linear.register_forward_pre_hook(
            lambda mod, args, kwargs, batcher=batcher: batcher.release((args, kwargs)),
            with_kwargs=True,
        )
        linear.register_forward_hook(
            lambda mod, args, kwargs, out, batcher=batcher: batcher.release(out),
            with_kwargs=True,
        )


class Requests:
    """The workers riding this engine's in-flight requests.

    Attributes:
        mediators: Worker by request id, for as long as the request lives.
        errored: Deferred error by request id for a payload that failed to
            deserialize — surfaced at collect instead of crashing the engine.
        rows: Request ids in the order the forward's tensors carry them.
        tokens: Token count each request contributes this step, by request id.
    """

    def __init__(self) -> None:
        self.mediators: dict[str, Any] = {}
        self.errored: dict[str, Any] = {}
        self.rows: list[str] = []
        self.tokens: dict[str, int] = {}

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

        interleaver = model.interleaver
        scheduled = []
        start = 0

        for request_id in self.rows:
            tokens = self.tokens.get(request_id)
            if tokens is None:
                continue
            mediator = self.mediators.get(request_id)
            if mediator is not None:
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
                if not finished or mediator.caches or mediator.exception is not None:
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
        interleaver = model.interleaver
        scheduled = {id(mediator) for mediator in interleaver.mediators}
        row = 0

        for request_id in self.rows:
            if self.tokens.get(request_id) is None:
                continue
            mediator = self.mediators.get(request_id)
            if mediator is not None and id(mediator) in scheduled:
                mediator.batch_group = [row, 1]
            row += 1

        interleaver.batcher.total = row

    def match(self, request_ids: set[str]) -> list[tuple[str, str]]:
        """Pair the engine's name for each request with this worker's.

        The two sides name the same request differently: vLLM appends a hash of the
        request's content on the way in, so a request the engine calls ``"0"``
        arrives here as ``"0-a2460f0e"``. Saves have to go home under the name the
        engine will recognize.

        Returns:
            ``(engine_id, worker_id)`` for each requested id that is ours.
        """
        matched = []
        for worker_id in self.mediators:
            engine_id = worker_id.rsplit("-", 1)[0]
            if engine_id in request_ids:
                matched.append((engine_id, worker_id))
            elif worker_id in request_ids:
                matched.append((worker_id, worker_id))
        return matched

    def record_saves(self) -> None:
        """Note, on each scheduled worker, which of its values were saved.

        Read from the thread-local save-set while still on the thread the workers
        ran on. Collection can happen on another thread — Ray dispatches it through
        its own RPC worker — where that thread-local is empty, so the answer is
        captured here and carried on the mediator instead. The set only grows across
        a request's steps, so re-recording each step keeps the latest superset.
        """
        from ....intervention.errors import capture_exception

        saved = _saves()
        for mediator in self.mediators.values():
            if mediator.batch_group is None:
                continue
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

        # pending is (event, location) for a read, (event, location, value) for a
        # swap/skip — index rather than unpack, or a worker parked on an edit crashes
        # here and takes the shared engine down with it.
        event, requester = mediator.pending[0], mediator.pending[1]
        if event is Event.BARRIER:
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

        batcher = VLLMBatcher()
        # With one rank nothing is sharded, so there is nothing to gather.
        sharded = get_tp_group().world_size > 1
        if sharded:
            _watch_parallel_linears(self.model, batcher)

        # An Envoy tree over the real module. Passing a loaded module builds it
        # directly, so no weights are read twice and the paths match the ones the
        # client's meta tree gave the user. Building it here also registers the
        # interleaver's hooks, which is what brackets the gather (see below).
        self.nnsight_model: VLLM = VLLM(self.model)
        self.nnsight_model.tokenizer = cached_tokenizer_from_config(self.model_config)

        if sharded:
            _release_parallel_linears(self.model, batcher)

        interleaver = self.nnsight_model.interleaver
        interleaver.mediators = []
        interleaver.batcher = batcher
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
        if not requests.mediators and not requests.errored:
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
        self, request_ids: list[str], finished_request_ids: Optional[list[str]] = None
    ) -> Optional[bytes]:
        """Return the saved values and any deferred error of the named requests.

        Keyed per request rather than merged, so two traces that happen to name a
        variable the same don't overwrite each other on the way home. Each entry is
        ``{"saves": {...}, "error": <deferred error or None>}``.

        Args:
            request_ids: Requests to collect saved values from.
            finished_request_ids: Those that are done, whose workers are wound up
                and forgotten afterwards.
        """
        # Only one rank holds the sampled output; the others have nothing to say.
        if get_pp_group().rank != 0:
            return None

        requests = self.nnsight_requests
        finished = set(finished_request_ids or [])
        matched = requests.match(set(request_ids) | finished)

        # A worker still parked when its request finishes was waiting on a location the
        # model never reached; surface that as its deferred error before it is read.
        for engine_id, worker_id in matched:
            if engine_id in finished:
                requests.finish_dangling(worker_id)

        collected = {
            engine_id: {
                "saves": requests.saves(worker_id),
                "error": requests.error(worker_id),
            }
            for engine_id, worker_id in matched
        }

        # Requests whose payload failed to deserialize (see Requests.add) carry no
        # worker; surface their captured error here, keyed like the collected saves.
        wanted = set(request_ids) | finished
        for req_id, error in list(requests.errored.items()):
            engine_id = req_id.rsplit("-", 1)[0]
            key = engine_id if engine_id in wanted else req_id if req_id in wanted else None
            if key is None:
                continue
            collected[key] = {"saves": {}, "error": error}
            if engine_id in finished or req_id in finished:
                requests.errored.pop(req_id, None)

        saved = _saves()
        for engine_id, worker_id in matched:
            if engine_id in finished:
                # Drop this request's saved values from the thread-local set as they
                # leave: it is keyed by object id, so a finished request's ids left
                # behind could be reused by a later request's values and mistaken for
                # saved. (No-op on a collect thread other than the workers' own, e.g.
                # Ray, where that set is empty — but harmless.)
                for value in collected[engine_id]["saves"].values():
                    saved.discard(id(value))
                requests.mediators.pop(worker_id, None)

        # Saves may still be device work in flight; they are about to be pickled.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return pickle.dumps(collected)

    def nnsight_request_count(self) -> int:
        """How many requests' workers this runner still tracks.

        A leak gauge: it should return to zero once every request has finished or
        been aborted. A number that only grows across requests means workers are
        outliving their requests — a finished one is freed in `collect_nnsight`,
        an aborted one when its stream is closed (see the async and serve backends).
        """
        return len(self.nnsight_requests.mediators)

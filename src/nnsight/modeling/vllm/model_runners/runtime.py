"""Intervention lifecycle shared by the vLLM model runner adapters."""

import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import zstandard as _zstd

from ....intervention.serialization import load
from ....intervention.tracing.globals import Globals

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData
    from ..vllm import VLLM
else:
    VLLM = Any

_ZSTD_COMPRESSOR = _zstd.ZstdCompressor(level=1)


class NNsightRequestHelper:
    """Own request mediators and trace-shared saves across scheduler steps."""

    def __init__(self):

        self.req_id_to_batch_group_idx: Dict[str, int] = {}
        self.mediators: Dict[str, Any] = {}  # req_id -> Mediator
        self.trace_contexts: Dict[str, dict] = {}  # trace_id -> context
        self._batch_req_ids: List[str] = []
        self._num_scheduled_tokens: Dict[str, int] = {}

    def process_new_reqs(
        self, new_reqs: List["NewRequestData"], model: VLLM
    ) -> None:
        """
        Process new requests and organize them into batch groups for execution.

        Each request carries its own serialized mediator. When multiple
        mediators belong to the same trace (identified by trace_id), the
        first arrival's ``__globals__`` become the canonical reference.
        Subsequent arrivals graft the saved variable entries from the
        canonical globals into their own ``__globals__``, so all mediators
        share the same Python objects for cross-invoke state.

        Args:
            new_reqs (List[NewRequestData]): List of new request data objects to process.
        """

        for new_req in new_reqs:

            # Resumed requests can be admitted again after preemption.
            if new_req.req_id in self.mediators:
                continue

            if new_req.sampling_params is None:
                continue
            extra_args = new_req.sampling_params.extra_args
            if not extra_args:
                continue

            trace_id = extra_args.get("nnsight_trace_id")
            if trace_id is None:
                # Non-NNsight request, skip
                continue

            mediator = load(
                extra_args["nnsight_mediator"],
                model._remoteable_persistent_objects(),
            )

            saved_names = extra_args.get("nnsight_saved_names", [])

            # First mediator for this trace: create context and register
            # its __globals__ as canonical for shared variable grafting.
            if trace_id not in self.trace_contexts:
                canonical_globals = mediator.intervention.__globals__

                for name in saved_names:
                    if name in canonical_globals:
                        Globals.saves.add(id(canonical_globals[name]))

                self.trace_contexts[trace_id] = {
                    "saved_names": saved_names,
                    "canonical_globals": canonical_globals,
                    "expected_count": extra_args.get("nnsight_expected_count", 1),
                    "received_count": 0,
                    "pending_req_ids": set(),
                }
            else:
                # Subsequent mediator: graft saved vars from canonical
                # globals so all mediators share the same Python objects.
                ctx = self.trace_contexts[trace_id]
                canonical = ctx["canonical_globals"]
                med_globals = mediator.intervention.__globals__
                for name in saved_names:
                    if name in canonical:
                        med_globals[name] = canonical[name]

            ctx = self.trace_contexts[trace_id]

            mediator.idx = len(model.interleaver.mediators)
            model.interleaver.mediators.append(mediator)
            mediator.start(model.interleaver)

            self.mediators[new_req.req_id] = mediator
            ctx["pending_req_ids"].add(new_req.req_id)
            ctx["received_count"] += 1

    def unflatten(self, model: VLLM):
        """Re-assign batch groups from token-level to prompt-level.

        After the forward pass, logits have one row per *scheduled
        request* (in ``batch_req_ids`` order).  We must walk the
        same ordering used by ``process_batch_groups`` so that each
        mediator's prompt-level index matches its row in the logits
        tensor — even when the batch contains non-NNsight requests
        or requests whose mediators have already finished.
        """

        batch_start = 0
        mediator_set = {id(m) for m in model.interleaver.mediators}

        for req_id in self._batch_req_ids:
            if self._num_scheduled_tokens.get(req_id) is None:
                continue

            mediator = self.mediators.get(req_id)

            if mediator is None or id(mediator) not in mediator_set:
                # Non-NNsight request or already-finished mediator —
                # still occupies a row in the logits tensor.
                batch_start += 1
                continue

            mediator.batch_group = [batch_start, 1]
            batch_start += 1
            model.interleaver.batcher.last_batch_group = mediator.batch_group

    def process_batch_groups(
        self,
        num_tokens_scheduled: Dict[str, int],
        batch_req_ids: List[str],
        model: VLLM,
    ) -> None:

        # Clear batch_group for all registered mediators first. Persistent
        # cache hooks read mediator.batch_group live on each forward pass,
        # so a mediator whose request isn't scheduled in this step must
        # report "None" rather than the stale value from an earlier step
        # (which would point out-of-range in the smaller current batch).
        for m in self.mediators.values():
            m.batch_group = None

        batch_start = 0

        mediators = []

        # Iterate in input_batch order (batch_req_ids) rather than
        # scheduler dict order, because input_batch.condense() and
        # _may_reorder_batch() can reorder requests after the scheduler
        # builds num_scheduled_tokens.  The model's tensors (including
        # sampled_token_ids) follow input_batch order.
        for req_id in batch_req_ids:

            num_tokens = num_tokens_scheduled.get(req_id)
            if num_tokens is None:
                continue

            mediator = self.mediators.get(req_id)

            if mediator is None:
                batch_start += num_tokens
                continue

            mediators.append(mediator)
            mediator.batch_group = [batch_start, num_tokens]

            batch_start += num_tokens

        if mediators:
            model.interleaver.batcher.last_batch_group = mediators[-1].batch_group
        else:
            model.interleaver.batcher.last_batch_group = None

        model.interleaver.mediators = mediators

    def match_req_ids(self, req_id_set: set) -> List[tuple]:
        """Match engine-reported request IDs to stored mediators.

        vLLM appends a hash suffix to request IDs (e.g. ``"0-abc123"``
        or ``"uuid-abc123"``).  This method strips the suffix with
        ``rsplit`` and falls back to an exact match.

        Returns:
            List of ``(base_id, mediator, internal_key)`` tuples.
        """
        matched = []
        for req_id, mediator in self.mediators.items():
            base_id = req_id.rsplit("-", 1)[0]
            if base_id in req_id_set:
                matched.append((base_id, mediator, req_id))
            elif req_id in req_id_set:
                matched.append((req_id, mediator, req_id))
        return matched

    def finalize_mediators(self, matched, finished_req_id_set, model: VLLM) -> set:
        """Run result handler and cancel finished mediators.

        Returns:
            Set of internal keys for mediators that were finalized.
        """
        finished_internal_keys = set()
        for base_id, mediator, internal_key in matched:
            if base_id not in finished_req_id_set:
                continue

            finished_internal_keys.add(internal_key)

            if mediator.alive:
                model.interleaver.mediators = [mediator]
                mediator.active = True
                mediator.batch_group = None
                with model.interleaver:
                    model.interleaver.handle("result", [base_id])
                    mediator.cancel(preserve_locals=True)
                    model.interleaver.handle()
            # Always remove persistent cache hooks when the request
            # finishes — even if the mediator thread died early (e.g.
            # intervention code was just tracer.cache(); nns.save(c)
            # with no blocking access). Otherwise hooks pile up on the
            # module and keep firing with stale batch_groups from dead
            # mediators.
            mediator.remove_hooks()

        return finished_internal_keys

    def collect_saves(self, matched, finished_internal_keys: set) -> tuple:
        """Collect saved values from mediator frames, namespaced per request.

        Gathers per-invoke saves from frame locals and trace-shared
        saves from canonical globals (only when a trace is fully done).

        Returns:
            ``(saves_by_req, removals)`` —
            ``saves_by_req`` is ``{base_id: {var_name: value}}`` so
            concurrent requests whose user code uses the same
            variable name (``logits``, ``x``, …) don't collide at
            the outer flat-dict layer.  The caller (engine / server)
            routes each sub-dict to the matching request output.
            ``removals`` is a list of ``id`` values to discard from
            ``Globals.saves`` after collection.
        """
        saves_by_req: dict = {}
        removals = []

        # Map internal_key -> base_id so trace-shared saves from a
        # finished internal_key can be attached to the right bucket.
        base_by_internal = {ik: b for b, _, ik in matched}

        for base_id, mediator, internal_key in matched:
            per_req = saves_by_req.setdefault(base_id, {})
            frame = mediator.info.frame
            for key, value in frame.f_locals.items():
                if id(value) in Globals.saves:
                    per_req[key] = value
                    if internal_key in finished_internal_keys:
                        removals.append(id(value))

        # Trace-shared saves: collect when ALL mediators for a trace
        # have been received AND completed.  Attach to the base_id
        # whose finishing triggered the trace completion.
        for internal_key in finished_internal_keys:
            owning_base = base_by_internal.get(internal_key, internal_key)
            for _, ctx in self.trace_contexts.items():
                if internal_key in ctx["pending_req_ids"]:
                    ctx["pending_req_ids"].discard(internal_key)
                    trace_fully_done = (
                        not ctx["pending_req_ids"]
                        and ctx["received_count"] == ctx["expected_count"]
                    )
                    if trace_fully_done:
                        canonical = ctx["canonical_globals"]
                        per_req = saves_by_req.setdefault(owning_base, {})
                        for name in ctx["saved_names"]:
                            if name in canonical:
                                value = canonical[name]
                                if id(value) in Globals.saves:
                                    per_req[name] = value
                                    removals.append(id(value))
                    break

        return saves_by_req, removals

    def cleanup_finished(self, finished_internal_keys: set, removals: list) -> None:
        """Clean up state for finished requests.

        Discards collected IDs from ``Globals.saves``, deletes
        completed trace contexts, and drops mediator entries.
        """
        for _id in removals:
            Globals.saves.discard(_id)

        done_traces = [
            tid
            for tid, ctx in self.trace_contexts.items()
            if (
                not ctx["pending_req_ids"]
                and ctx["received_count"] == ctx["expected_count"]
            )
        ]
        for tid in done_traces:
            del self.trace_contexts[tid]

        for internal_key in finished_internal_keys:
            self.mediators.pop(internal_key, None)

    def discard_all(self):
        """Release uncollected interventions when the worker shuts down."""
        for mediator in self.mediators.values():
            mediator.cancel(preserve_locals=True)
            mediator.remove_hooks()
            Globals.saves.difference_update(
                id(value) for value in mediator.info.frame.f_locals.values()
            )
        for ctx in self.trace_contexts.values():
            canonical = ctx["canonical_globals"]
            Globals.saves.difference_update(
                id(canonical[name])
                for name in ctx["saved_names"]
                if name in canonical
            )
        self.mediators.clear()
        self.trace_contexts.clear()


class NNsightRunnerMixin:
    NNsightRequestHelper = NNsightRequestHelper

    def __init__(self, *args, **kwargs):

        from .. import VLLM

        super().__init__(*args, **kwargs)

        self.nnsight_model: Optional[VLLM] = None

        self.nnsight_request_helper = self.NNsightRequestHelper()

    def load_model(self, *args, **kwargs) -> None:

        from .. import VLLM
        from ..batching import VLLMBatcher
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.tokenizers import cached_tokenizer_from_config

        super().load_model(*args, **kwargs)

        self.nnsight_model = VLLM(self.model)

        self.nnsight_model.tokenizer = cached_tokenizer_from_config(self.model_config)

        self.nnsight_model.interleaver.mediators = []

        self.nnsight_model.interleaver.batcher = VLLMBatcher()

        self.nnsight_model.interleaver.defer_exceptions = True

        # Mount ``Object.save`` in the worker subprocess. The driver
        # process gets its mount via ``Tracer.__init__``/``__setstate__``
        # (the client constructs Tracers, the serve handler unpickles
        # them), but vLLM workers receive only deserialized Mediators —
        # they never touch a Tracer. Without this call, user code's
        # ``tensor.save()`` AttributeErrors inside the worker thread.
        from ....intervention.tracing.globals import _ensure_mounted

        _ensure_mounted()

        # Only wrap when TP > 1: registers hooks that handle
        # gather/split of sharded tensors and CUDA synchronization
        # for TP-parallel modules.  With TP == 1 nothing is sharded
        # so wrapping is pure overhead.

        if get_tp_group().world_size > 1:
            self.nnsight_model.interleaver.batcher.wrap(self.nnsight_model)

    def collect_nnsight(
        self,
        req_ids: list[str],
        finished_req_ids: list[str] | None = None,
    ) -> Optional[bytes]:
        """Collect saved values from mediators, optionally finalizing finished requests.

        Called on every streamed output (async) or on finished requests (sync).
        Saves are collected for ALL ``req_ids``.  Mediators listed in
        ``finished_req_ids`` are additionally finalized (result handler,
        cancel) and cleaned up.

        Args:
            req_ids: Request IDs to collect current saves from.
            finished_req_ids: Subset of request IDs that are finished and
                should be finalized and cleaned up.  ``None`` means no
                requests are finished.
        """
        from vllm.distributed.parallel_state import get_pp_group

        if get_pp_group().rank != 0:
            return None

        if finished_req_ids is None:
            finished_req_ids = []

        helper = self.nnsight_request_helper
        req_id_set = set(req_ids) | set(finished_req_ids)
        finished_req_id_set = set(finished_req_ids)

        matched = helper.match_req_ids(req_id_set)
        finished_keys = helper.finalize_mediators(
            matched, finished_req_id_set, self.nnsight_model
        )
        saves_by_req, removals = helper.collect_saves(matched, finished_keys)
        helper.cleanup_finished(finished_keys, removals)

        # Collect deferred exceptions.  Each mediator has its own —
        # nest the typed envelope inside THAT request's sub-dict under
        # ``__nnsight_exceptions__`` in the ``{base_id: DeferredError}``
        # shape the client's ``surface_server_errors`` understands.
        # The dynamic NNsightException subclass can't be pickled, so the
        # helper ships plain strings (type_name, message, traceback,
        # is_control_flow).
        from ....intervention.errors import capture_deferred

        for base_id, mediator, _internal_key in matched:
            entry = capture_deferred(mediator, req_id=base_id)
            if entry is not None:
                per_req = saves_by_req.setdefault(base_id, {})
                per_req["__nnsight_exceptions__"] = {base_id: entry}
                mediator.deferred_exception = None
                mediator._deferred_type_name = None
                mediator._deferred_traceback = None
                mediator._deferred_is_control_flow = False

        torch.cuda.synchronize()
        return _ZSTD_COMPRESSOR.compress(pickle.dumps(saves_by_req))

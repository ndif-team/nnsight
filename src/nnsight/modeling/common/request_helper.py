"""Shared request helper for continuous batching backends (vLLM, HF CB).

This module provides ``NNsightRequestHelper``, which manages the mapping
between engine requests and nnsight mediators.  It is engine-agnostic:
both vLLM and HF CB use the same helper, differing only in how mediators
arrive (serialized vs direct) and which dimension the batcher slices
(dim 0 for vLLM's 2D tensors, dim 1 for HF CB's 3D tensors).

The helper tracks:
- ``mediators``: req_id → Mediator mapping
- ``trace_contexts``: trace_id → shared state for cross-invoke variable grafting

Save lifecycle: saves are kept in the process-wide ``Globals.saves``
set. Collection iterates each mediator's own ``info.frame.f_locals``
and filters by ``id(v) in Globals.saves`` — the per-frame ownership
boundary keeps concurrent requests isolated even though the id set is
shared. Cleanup is per-id (``Globals.saves.discard``); the whole set
is never cleared on the server side.
"""

from typing import Any, Dict, List, Optional, Set

from ...intervention.tracing.globals import Globals


class NNsightRequestHelper:
    """Manages mediator ↔ request mapping for continuous batching backends.

    This class is shared between vLLM and HF CB.  Engine-specific code
    uses one of the ``process_new_reqs_*`` methods to register mediators,
    then calls the shared methods for batch group assignment, finalization,
    save collection, and cleanup.
    """

    def __init__(self):
        self.mediators: Dict[str, Any] = {}  # req_id -> Mediator
        self.trace_contexts: Dict[str, dict] = {}  # trace_id -> context
        # vLLM-only: ``unflatten()`` reads these to map per-row logits
        # back to per-request mediator slices. Set by
        # ``GPUModelRunner._update_states`` before each forward pass and
        # cleared by ``_fail_scheduled`` on the HF vanilla path
        # (defensively — no HF path actually calls ``unflatten``).
        # Initialised here so a fresh helper or a call before the first
        # batch step is a clean no-op rather than an ``AttributeError``.
        self._batch_req_ids: List[str] = []
        self._num_scheduled_tokens: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Mediator registration (engine-specific entry points)
    # ------------------------------------------------------------------

    def _register_mediator(
        self,
        req_id: str,
        mediator: Any,
        trace_id: str,
        saved_names: List[str],
        expected_count: int,
        model: Any,
    ) -> None:
        """Pure bookkeeping for a new mediator.

        Sets up the trace context, grafts canonical globals for
        cross-invoke variable sharing, pre-registers saved-name IDs in
        ``Globals.saves`` so trace-shared variables survive the
        :meth:`collect_saves` filter, and records the mediator in the
        helper's dict. Does NOT start the worker or touch the
        interleaver's ``mediators`` list.

        Start policy is the caller's responsibility:

        - Callers that register while an interleaver is already active
          (vLLM: ``_update_states`` runs inside ``with interleaver:``,
          so ``__enter__`` has already run and won't pick up new
          mediators) must call :meth:`_start_mediator_now` immediately
          after this method.
        - Callers that register before an interleaver is entered (HF
          CB vanilla, and the paged HF path when completed: scheduler
          runs before ``_step()`` enters the interleaver) can rely on
          ``Interleaver.__enter__``'s auto-start loop.
        """
        # First mediator for this trace: create context and register
        # its __globals__ as canonical for shared variable grafting.
        if trace_id not in self.trace_contexts:
            canonical_globals = mediator.intervention.__globals__

            # Pre-register saved-name IDs in the process-wide
            # ``Globals.saves`` set so trace-shared canonical-globals
            # values survive ``collect_saves``'s id filter. Matches the
            # pattern dev's inline ``NNsightRequestHelper.process_new_reqs``
            # uses in ``GPUModelRunner.py``.
            for name in saved_names:
                if name in canonical_globals:
                    Globals.saves.add(id(canonical_globals[name]))

            self.trace_contexts[trace_id] = {
                "saved_names": saved_names,
                "canonical_globals": canonical_globals,
                "expected_count": expected_count,
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
        self.mediators[req_id] = mediator
        ctx["pending_req_ids"].add(req_id)
        ctx["received_count"] += 1

    def _start_mediator_now(self, mediator: Any, model: Any) -> None:
        """Start a mediator's worker immediately.

        Appropriate only when called from inside an already-entered
        ``with model.interleaver:`` block (``_interleaving`` is True).
        """
        model.interleaver.mediators.append(mediator)
        mediator.start(model.interleaver)

    def process_new_reqs_serialized(
        self, new_reqs: list, model: Any
    ) -> None:
        """vLLM path: deserialize mediators from sampling params extra_args,
        register them, and start their workers immediately.

        Called from ``GPUModelRunner._update_states`` inside a live
        ``with interleaver:`` block — ``__enter__`` has already run
        and won't auto-start newly registered mediators, so we start
        them inline after registration.

        Args:
            new_reqs: List of vLLM ``NewRequestData`` objects.
            model: The NNsight-wrapped model (e.g. ``VLLM`` instance).
        """
        from ...intervention.serialization import load

        for new_req in new_reqs:
            extra_args = getattr(new_req.sampling_params, "extra_args", None)
            if not extra_args:
                continue

            trace_id = extra_args.get("nnsight_trace_id")
            if trace_id is None:
                continue

            mediator = load(
                extra_args["nnsight_mediator"],
                model._remoteable_persistent_objects(),
            )
            saved_names = extra_args.get("nnsight_saved_names", [])
            expected_count = extra_args.get("nnsight_expected_count", 1)

            self._register_mediator(
                req_id=new_req.req_id,
                mediator=mediator,
                trace_id=trace_id,
                saved_names=saved_names,
                expected_count=expected_count,
                model=model,
            )
            self._start_mediator_now(mediator, model)

    def process_new_reqs_direct(
        self, entries: List[tuple], model: Any
    ) -> None:
        """HF CB path: receive mediators directly (no serialization).

        Registers bookkeeping only; does NOT start workers. HF CB's
        scheduler runs before ``_step()`` enters the interleaver, so
        ``Interleaver.__enter__`` picks up newly-registered mediators
        via its auto-start loop.

        Args:
            entries: List of ``(req_id, mediator, trace_id, saved_names,
                expected_count)`` tuples.
            model: The NNsight-wrapped model (e.g. ``LanguageModel`` instance).
        """
        for req_id, mediator, trace_id, saved_names, expected_count in entries:
            self._register_mediator(
                req_id=req_id,
                mediator=mediator,
                trace_id=trace_id,
                saved_names=saved_names,
                expected_count=expected_count,
                model=model,
            )

    # ------------------------------------------------------------------
    # Batch group assignment (shared — both backends pack tokens the same way)
    # ------------------------------------------------------------------

    def process_batch_groups(
        self,
        num_tokens_scheduled: Dict[str, int],
        batch_req_ids: List[str],
        model: Any,
    ) -> None:
        """Assign ``mediator.batch_group = [token_start, num_tokens]`` per step.

        Both vLLM and HF CB pack tokens contiguously.  The batch group
        offsets are identical; the batcher's ``batch_dim`` parameter
        determines which tensor dimension to slice (dim 0 for vLLM's 2D
        tensors, dim 1 for HF CB's 3D tensors).

        Args:
            num_tokens_scheduled: Mapping of req_id → number of tokens
                scheduled this step.
            batch_req_ids: Request IDs in batch order (determines tensor
                layout after any reordering by the engine).
            model: The NNsight-wrapped model.
        """
        batch_start = 0
        mediators = []

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

    def unflatten(self, model: Any) -> None:
        """Re-assign batch groups from token-level to prompt-level.

        After the forward pass, logits have one row per *scheduled
        request* (in ``batch_req_ids`` order).  We walk the same
        ordering used by ``process_batch_groups`` so that each
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
                batch_start += 1
                continue

            mediator.batch_group = [batch_start, 1]
            batch_start += 1
            model.interleaver.batcher.last_batch_group = mediator.batch_group

    # ------------------------------------------------------------------
    # Request ID matching
    # ------------------------------------------------------------------

    def match_req_ids(
        self, req_id_set: set, strip_suffix: bool = True
    ) -> List[tuple]:
        """Match engine-reported request IDs to stored mediators.

        Args:
            req_id_set: Set of request IDs reported by the engine.
            strip_suffix: If True, strip hash suffix from req IDs
                (vLLM appends ``"-abc123"``).  HF CB should pass False.

        Returns:
            List of ``(base_id, mediator, internal_key)`` tuples.
        """
        matched = []
        for req_id, mediator in self.mediators.items():
            if strip_suffix:
                base_id = req_id.rsplit("-", 1)[0]
                if base_id in req_id_set:
                    matched.append((base_id, mediator, req_id))
                elif req_id in req_id_set:
                    matched.append((req_id, mediator, req_id))
            else:
                if req_id in req_id_set:
                    matched.append((req_id, mediator, req_id))
        return matched

    # ------------------------------------------------------------------
    # Finalization and save collection (shared)
    # ------------------------------------------------------------------

    def finalize_mediators(
        self, matched: list, finished_req_id_set: set, model: Any
    ) -> set:
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
                mediator.batch_group = None
                with model.interleaver:
                    model.interleaver.handle("result", [base_id])
                    mediator.cancel()
                    model.interleaver.handle()

        return finished_internal_keys

    def collect_saves(
        self, matched: list, finished_internal_keys: set
    ) -> tuple:
        """Collect saved values from mediator frames.

        Filters each mediator's ``info.frame.f_locals`` against the
        process-wide ``Globals.saves`` set. Per-frame iteration keeps
        concurrent requests isolated: even though the id set is shared,
        request B never iterates request A's frame, so A's ids are
        never observed by B's filter pass.

        Returns:
            ``(saves, removals)`` — the saves dict and list of ``id()``
            values to discard from ``Globals.saves`` in
            :meth:`cleanup_finished`.
        """
        saves = {}
        removals = []

        for base_id, mediator, internal_key in matched:
            frame = mediator.info.frame
            for key, value in frame.f_locals.items():
                if id(value) in Globals.saves:
                    saves[key] = value
                    if internal_key in finished_internal_keys:
                        removals.append(id(value))

        # Trace-shared saves: collect when ALL mediators for a trace
        # have been received AND completed.
        for internal_key in finished_internal_keys:
            for _, ctx in self.trace_contexts.items():
                if internal_key in ctx["pending_req_ids"]:
                    ctx["pending_req_ids"].discard(internal_key)
                    trace_fully_done = (
                        not ctx["pending_req_ids"]
                        and ctx["received_count"] == ctx["expected_count"]
                    )
                    if trace_fully_done:
                        canonical = ctx["canonical_globals"]
                        for name in ctx["saved_names"]:
                            if name in canonical:
                                value = canonical[name]
                                if id(value) in Globals.saves:
                                    saves[name] = value
                                    removals.append(id(value))
                    break

        return saves, removals

    def cleanup_finished(
        self, finished_internal_keys: set, removals: list
    ) -> None:
        """Clean up state for finished requests.

        Discards collected IDs from ``Globals.saves``, deletes completed
        trace contexts, and drops mediator entries.
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

"""Regression tests for C5: forward-failure blast radius and state leak.

Two invariants asserted:

1. A forward failure scopes to the ``scheduled`` batch only. Requests
   in ``self._active`` but not in ``scheduled`` are untouched — they
   keep their signals, keep their state, and can continue running in
   subsequent steps. Pre-fix: the ``_generation_loop`` catch-all
   iterated ``self._active.keys()`` and finalized every active
   request regardless of whether it participated in the failing
   batch (Problem A).

2. After a forward failure, per-step shared state is clean:
   ``helper.mediators`` contains no stale entries for the failed
   batch, ``interleaver.mediators`` is empty, ``_batch_req_ids`` and
   ``_num_scheduled_tokens`` are reset. Pre-fix: state leaked,
   accumulating dict entries per failed request over the server's
   lifetime (Problem B memory leak).
"""

from __future__ import annotations

import threading

import pytest
import torch
from transformers import DynamicCache, GenerationConfig


@pytest.fixture(scope="module")
def model():
    from nnsight import LanguageModel
    m = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    yield m


class TestFailureScope:
    def test_failure_scopes_to_scheduled_batch_not_active(self, model):
        """Directly construct ``_active={r0,r1,r2,r3}``, ``scheduled={r0,r1}``,
        make ``_step`` fail, and verify only the scheduled batch is tanked.

        Natural reproduction via the scheduler is difficult because
        decode-first priority keeps ``_active == scheduled`` in steady
        state. Direct state construction is the targeted way to
        exercise the blast-radius code path.
        """
        from nnsight.modeling.hf_serve.vanilla_server import (
            VanillaBatchServer, ActiveRequest, ScheduledItem,
        )

        server = VanillaBatchServer(model, token_budget=8, max_batch_size=8)
        server.request_helper.process_new_reqs_direct = lambda *a, **kw: None
        server.request_helper.mediators = {}

        signals = {}
        for rid in ["r0", "r1", "r2", "r3"]:
            active = ActiveRequest(
                req_id=rid,
                prompt_ids=[1, 2, 3],
                generated_ids=[1, 2, 3],
                max_new_tokens=10,
                eos_token_ids=set(),
                past_key_values=DynamicCache(),
                prefilled_len=3,
                cache_mask=[1, 1, 1],
            )
            server._active[rid] = active
            event = threading.Event()
            signals[rid] = event
            server._result_signals[rid] = event

        def failing_step(scheduled):
            raise RuntimeError("simulated forward failure")

        server._step = failing_step

        scheduled = [
            ScheduledItem(
                request=server._active["r0"],
                num_tokens=1, is_prefill=False, token_ids=[1],
            ),
            ScheduledItem(
                request=server._active["r1"],
                num_tokens=1, is_prefill=False, token_ids=[1],
            ),
        ]

        # Drive _step_with_rollback directly — mirrors what
        # _generation_loop does.
        server._step_with_rollback(scheduled)

        # Scheduled batch: finalized with __error__, signals set,
        # popped from _active.
        assert signals["r0"].is_set(), "r0 was in scheduled batch — must be finalized"
        assert signals["r1"].is_set(), "r1 was in scheduled batch — must be finalized"
        result_r0 = server.get_result("r0")
        result_r1 = server.get_result("r1")
        assert result_r0 and "__error__" in result_r0
        assert result_r1 and "__error__" in result_r1
        assert result_r0["__error__"]["type_name"] == "RuntimeError"
        assert result_r0["__error__"]["req_id"] == "r0"

        # Unscheduled-but-active: signals NOT set, requests STILL in _active.
        assert not signals["r2"].is_set(), (
            "r2 was NOT in scheduled batch — must not be finalized. "
            "Pre-fix bug: catch-all tanked all _active indiscriminately."
        )
        assert not signals["r3"].is_set(), (
            "r3 was NOT in scheduled batch — must not be finalized."
        )
        assert "r2" in server._active, "r2 must remain in _active"
        assert "r3" in server._active, "r3 must remain in _active"


class TestStateCleanupAfterFailure:
    def test_helper_and_interleaver_state_clean_after_failure(self, model):
        """After a forward failure, per-step shared state must be reset.

        Uses a real trace (mediator path) so the state-writes at
        ``_step:642-671`` actually run before the failure — exercising
        the exact leak path the fix closes.
        """
        from nnsight.intervention.backends import Backend
        from nnsight.intervention.tracing.globals import Globals
        from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

        server = VanillaBatchServer(model, token_budget=128, max_batch_size=4)

        # Patch _step to fail AFTER the state-writes have happened.
        # Easiest way: replace model._model's forward via a side channel
        # that raises exactly once. This keeps the state-write path
        # live (they happen before the forward call).
        original_step = server._step
        fail_once = {"armed": False}

        def failing_on_forward_step(scheduled):
            if fail_once["armed"]:
                fail_once["armed"] = False
                # Hand-craft the same shared-state writes that _step does
                # before its forward, then raise. This puts the leak
                # hazard on display without needing to mock model._model.
                helper = server.request_helper
                m = server.model
                batch_req_ids = [item.request.req_id for item in scheduled]
                num_tokens_map = {item.request.req_id: 1 for item in scheduled}
                if any(helper.mediators.get(r) is not None for r in batch_req_ids):
                    helper.process_batch_groups(
                        num_tokens_map, batch_req_ids, m,
                    )
                    m.interleaver.batcher.needs_batching = len(scheduled) > 1
                    helper._batch_req_ids = batch_req_ids
                    helper._num_scheduled_tokens = num_tokens_map
                raise RuntimeError("simulated forward failure after state writes")
            return original_step(scheduled)

        server._step = failing_on_forward_step

        class CapturingBackend(Backend):
            def __init__(self_inner, srv):
                self_inner.server = srv

            def __call__(self_inner, tracer):
                if tracer is None:
                    return
                interventions = Backend.__call__(self_inner, tracer)
                try:
                    Globals.enter()
                    _args, kwargs = tracer._run_user_fn(interventions)
                    # Test simulator: restore interleaver state for bg thread
                    # since the model fixture is shared across tests.
                    tracer._init_shared_interleaver()
                    entries = self_inner.server.build_entries(kwargs, mediators=tracer.mediators)
                    tracer.mediators.clear()
                    pending = [
                        (e, self_inner.server.submit(e)) for e in entries
                    ]
                finally:
                    Globals.exit()
                # Arm the failure for the step that processes these entries.
                fail_once["armed"] = True
                for _, event in pending:
                    event.wait(timeout=10.0)

        backend = CapturingBackend(server)

        server.start()
        try:
            with model.trace("Hello", backend=backend):
                out = model.lm_head.output.save()
        finally:
            server.stop()
            # Reset model interleaver so other tests aren't polluted.
            iv = model.interleaver
            for m in list(iv.mediators or []):
                try:
                    m.cancel()
                except Exception:
                    pass
            iv.mediators = []

        # Post-fix invariants: all state fields are clean.
        helper = server.request_helper
        iv = server.model.interleaver

        assert helper.mediators == {}, (
            f"helper.mediators leaked entries after failure: "
            f"{list(helper.mediators.keys())}"
        )
        assert iv.mediators == [], (
            f"interleaver.mediators leaked after failure: {iv.mediators}"
        )
        assert helper._batch_req_ids == [], (
            f"_batch_req_ids not reset: {helper._batch_req_ids}"
        )
        assert helper._num_scheduled_tokens == {}, (
            f"_num_scheduled_tokens not reset: {helper._num_scheduled_tokens}"
        )
        assert iv.batcher.last_batch_group is None, (
            f"batcher.last_batch_group not reset: {iv.batcher.last_batch_group}"
        )
        assert iv.batcher.needs_batching is False, (
            f"batcher.needs_batching not reset: {iv.batcher.needs_batching}"
        )


class TestBgLoopSafetyNet:
    def test_unexpected_exception_does_not_tank_all_active(self, model):
        """The bg-loop catch-all is a safety net now — an exception
        escaping ``_step_with_rollback`` must not tank every active
        request indiscriminately. The safety net should scope to
        whatever ``scheduled`` was visible.
        """
        from nnsight.modeling.hf_serve.vanilla_server import (
            VanillaBatchServer, ActiveRequest, ScheduledItem,
        )

        server = VanillaBatchServer(model, token_budget=8, max_batch_size=8)
        server.request_helper.process_new_reqs_direct = lambda *a, **kw: None
        server.request_helper.mediators = {}

        signals = {}
        for rid in ["a", "b", "c", "d"]:
            active = ActiveRequest(
                req_id=rid,
                prompt_ids=[1, 2], generated_ids=[1, 2],
                max_new_tokens=5, eos_token_ids=set(),
                past_key_values=DynamicCache(),
                prefilled_len=2, cache_mask=[1, 1],
            )
            server._active[rid] = active
            event = threading.Event()
            signals[rid] = event
            server._result_signals[rid] = event

        # Simulate the catch-all escape path: patch
        # ``_step_with_rollback`` to raise instead of handling. This is
        # the "should never fire" branch in _generation_loop's
        # safety-net try/except.
        def escaping_rollback(scheduled):
            raise RuntimeError("exception escaped _step_with_rollback")

        server._step_with_rollback = escaping_rollback

        # Directly execute the body of _generation_loop's
        # try/except for one iteration. We're not spinning the bg
        # thread because we want deterministic inspection.
        scheduled = [
            ScheduledItem(
                request=server._active["a"],
                num_tokens=1, is_prefill=False, token_ids=[1],
            ),
            ScheduledItem(
                request=server._active["b"],
                num_tokens=1, is_prefill=False, token_ids=[1],
            ),
        ]

        try:
            server._step_with_rollback(scheduled)
        except Exception as e:
            server._fail_scheduled(scheduled, e)

        # Safety-net scoped to scheduled — c, d untouched.
        assert signals["a"].is_set()
        assert signals["b"].is_set()
        assert not signals["c"].is_set(), (
            "c was not in scheduled batch — safety net must NOT tank it"
        )
        assert not signals["d"].is_set(), (
            "d was not in scheduled batch — safety net must NOT tank it"
        )

"""Tests for vanilla batch server and HF continuous batching integration.

Run with:
    conda activate ndif-dev
    pytest tests/test_hf_serve.py -x -v
"""

import pytest
import torch

# Skip CB-specific tests if transformers doesn't have CB support
try:
    from transformers.generation.continuous_batching import ContinuousBatchingManager
    HAS_CB = True
except ImportError:
    HAS_CB = False


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Load GPT-2 as LanguageModel — same as users would."""
    from nnsight import LanguageModel
    m = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    yield m


# ------------------------------------------------------------------
# LanguageModel trace tests (baseline — these always worked)
# ------------------------------------------------------------------

class TestLanguageModelTrace:
    """Verify LanguageModel trace works as expected (sanity baseline)."""

    def test_trace_forward_pass(self, model):
        with model.trace("Hello"):
            output = model.lm_head.output.save()
        assert output.dim() == 3  # [batch, seq, vocab]

    def test_hidden_states_standard_shape(self, model):
        # transformers 5.x: GPT2Block.forward returns the hidden_states
        # tensor directly (3-D), not the legacy (hidden_states, ...) tuple.
        with model.trace("Hello"):
            hs = model.transformer.h[0].output.save()
        assert hs.dim() == 3
        assert hs.shape[0] == 1
        assert hs.shape[-1] == 768

    def test_trace_matches_direct_forward(self, model):
        with model.trace("Hello world"):
            output_trace = model.lm_head.output.save()

        tokens = model.tokenizer("Hello world", return_tensors="pt")
        input_ids = tokens["input_ids"].to(model.device)
        with torch.no_grad():
            direct_out = model._model(input_ids=input_ids)

        assert output_trace.shape == direct_out.logits.shape
        assert torch.allclose(output_trace, direct_out.logits, atol=1e-5)


# ------------------------------------------------------------------
# Vanilla batch server tests
# ------------------------------------------------------------------

class TestVanillaServer:
    """Test VanillaBatchServer wrapping a LanguageModel."""

    def test_import(self):
        from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer, VanillaRequest
        assert VanillaBatchServer is not None
        assert VanillaRequest is not None

    def test_vanilla_request_dataclass(self):
        from nnsight.modeling.hf_serve.vanilla_server import VanillaRequest
        req = VanillaRequest(
            req_id="test_0",
            token_ids=[1, 2, 3],
            gen_kwargs={"max_new_tokens": 5},
            mediator=None,
            trace_id="trace_0",
            saved_names=[],
            expected_count=1,
        )
        assert req.req_id == "test_0"
        assert req.token_ids == [1, 2, 3]

    def test_server_lifecycle(self, model):
        from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer
        server = VanillaBatchServer(model, max_batch_size=4)
        assert not server.is_running()
        server.start()
        assert server.is_running()
        server.stop()
        assert not server.is_running()

    def test_server_wraps_language_model(self, model):
        """Server should accept a LanguageModel instance."""
        from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer
        server = VanillaBatchServer(model)
        assert server.model is model
        assert server.request_helper is not None

    def test_scheduler_token_budget(self, model):
        """Scheduler should respect the token budget."""
        from nnsight.modeling.hf_serve.vanilla_server import (
            VanillaBatchServer, VanillaRequest, ActiveRequest,
        )
        from transformers import DynamicCache

        server = VanillaBatchServer(model, token_budget=10)

        # Add a request with 25 prompt tokens (exceeds budget of 10)
        req = VanillaRequest(
            req_id="test_0", token_ids=list(range(25)),
            gen_kwargs={"max_new_tokens": 5}, mediator=None,
            trace_id="t0", saved_names=[], expected_count=1,
        )
        server._pending.append(req)

        # Mock: skip mediator registration (no real mediator)
        original_activate = server._activate_request
        def mock_activate(vr):
            eos_id = getattr(model._model.config, "eos_token_id", None)
            if isinstance(eos_id, list):
                eos_id = eos_id[0]
            active = ActiveRequest(
                req_id=vr.req_id, prompt_ids=vr.token_ids,
                generated_ids=[], max_new_tokens=5,
                eos_token_id=eos_id or -1,
                past_key_values=DynamicCache(),
                prefilled_len=0, cache_mask=[],
            )
            server._active[vr.req_id] = active
            return active
        server._activate_request = mock_activate

        scheduled = server._schedule()
        assert len(scheduled) == 1
        # Should chunk: only 10 tokens (the budget), not all 25
        assert scheduled[0].num_tokens == 10
        assert scheduled[0].is_prefill is True
        assert scheduled[0].token_ids == list(range(10))

    def test_scheduler_decode_first(self, model):
        """Decode requests should be scheduled before prefill."""
        from nnsight.modeling.hf_serve.vanilla_server import (
            VanillaBatchServer, VanillaRequest, ActiveRequest,
        )
        from transformers import DynamicCache

        server = VanillaBatchServer(model, token_budget=5)

        # Add a decoding request (already prefilled)
        decode_req = ActiveRequest(
            req_id="decode_0", prompt_ids=[1, 2, 3],
            generated_ids=[4], max_new_tokens=10,
            eos_token_id=-1, past_key_values=DynamicCache(),
            prefilled_len=3, cache_mask=[1, 1, 1, 1],
        )
        server._active["decode_0"] = decode_req

        # Add a pending prefill request
        pending = VanillaRequest(
            req_id="prefill_0", token_ids=list(range(20)),
            gen_kwargs={"max_new_tokens": 5}, mediator=None,
            trace_id="t1", saved_names=[], expected_count=1,
        )
        server._pending.append(pending)

        def mock_activate(vr):
            active = ActiveRequest(
                req_id=vr.req_id, prompt_ids=vr.token_ids,
                generated_ids=[], max_new_tokens=5,
                eos_token_id=-1, past_key_values=DynamicCache(),
                prefilled_len=0, cache_mask=[],
            )
            server._active[vr.req_id] = active
            return active
        server._activate_request = mock_activate

        scheduled = server._schedule()
        # Decode first (1 token), then prefill gets remaining 4 tokens
        assert len(scheduled) == 2
        assert scheduled[0].request.req_id == "decode_0"
        assert scheduled[0].num_tokens == 1
        assert scheduled[0].is_prefill is False
        assert scheduled[1].request.req_id == "prefill_0"
        assert scheduled[1].num_tokens == 4  # budget=5 minus 1 decode = 4
        assert scheduled[1].is_prefill is True


# ------------------------------------------------------------------
# Batcher tests
# ------------------------------------------------------------------

class TestHFBatcher:
    """Test HFBatcher narrow/swap on dim 1."""

    def test_narrow_dim1(self):
        from nnsight.modeling.hf_serve.batching import HFBatcher
        b = HFBatcher(batch_dim=1)
        b.needs_batching = True
        b.last_batch_group = [3, 2]  # total = 5

        b.current_value = torch.randn(1, 5, 768)
        result = b.narrow([0, 3])
        assert result.shape == (1, 3, 768)

        result2 = b.narrow([3, 2])
        assert result2.shape == (1, 2, 768)

    def test_narrow_no_batching(self):
        from nnsight.modeling.hf_serve.batching import HFBatcher
        b = HFBatcher(batch_dim=1)
        b.needs_batching = False
        b.current_value = torch.randn(1, 5, 768)
        result = b.narrow([0, 3])
        assert result.shape == (1, 5, 768)  # full tensor returned

    def test_swap_dim1(self):
        from nnsight.modeling.hf_serve.batching import HFBatcher
        b = HFBatcher(batch_dim=1)
        b.needs_batching = True
        b.last_batch_group = [3, 2]

        b.current_value = torch.zeros(1, 5, 4)
        new_val = torch.ones(1, 2, 4)
        b.swap([3, 2], new_val)

        assert torch.all(b.current_value[0, :3, :] == 0)
        assert torch.all(b.current_value[0, 3:, :] == 1)

    def test_swap_full_batch(self):
        from nnsight.modeling.hf_serve.batching import HFBatcher
        b = HFBatcher(batch_dim=1)
        b.needs_batching = False
        b.current_value = torch.zeros(1, 5, 4)
        new_val = torch.ones(1, 5, 4)
        b.swap(None, new_val)
        assert torch.all(b.current_value == 1)

    def test_total_batch_size(self):
        from nnsight.modeling.hf_serve.batching import HFBatcher
        b = HFBatcher(batch_dim=1)
        b.last_batch_group = [10, 5]
        assert b.total_batch_size == 15

    def test_total_batch_size_none(self):
        from nnsight.modeling.hf_serve.batching import HFBatcher
        b = HFBatcher(batch_dim=1)
        assert b.total_batch_size == 0


# ------------------------------------------------------------------
# NNsightCBManager (paged HF CB) mediator registration wiring
# ------------------------------------------------------------------

# Skip paged HF CB tests if the installed transformers doesn't
# expose `ContinuousBatchingConfig` (manager.py imports fail). The
# paged path is feature-gated on HF's continuous-batching API surface
# existing; vanilla path is unaffected.
_paged_skip_reason = None
try:
    from nnsight.modeling.hf_serve.manager import NNsightCBManager  # noqa: F401
except Exception as _e:
    _paged_skip_reason = f"paged HF CB unavailable: {_e}"


@pytest.mark.skipif(
    _paged_skip_reason is not None,
    reason=_paged_skip_reason or "",
)
class TestNNsightCBManagerRegistration:
    """Paged HF CB path: `add_request` stashes mediator data and
    `_register_pending_mediators` calls `process_new_reqs_direct` for
    IDs that enter the batch.

    These tests exercise the registration wiring in isolation (no
    actual forward pass — HF's paged CB needs a real HF
    PreTrainedModel and scheduler setup that's beyond a unit test).
    """

    def test_add_request_stashes_mediator_data(self):
        from nnsight.modeling.hf_serve.manager import NNsightCBManager

        # Skip HF base __init__ — we only exercise the nnsight-layer bookkeeping.
        mgr = NNsightCBManager.__new__(NNsightCBManager)
        mgr._pending_nnsight_data = {}

        # Patch super().add_request to bypass HF init path
        def fake_super_add(input_ids, request_id=None, max_new_tokens=None):
            return request_id or "req_auto"
        import nnsight.modeling.hf_serve.manager as mgr_mod
        original = mgr_mod.ContinuousBatchingManager.add_request
        mgr_mod.ContinuousBatchingManager.add_request = (
            lambda self, input_ids, request_id=None, max_new_tokens=None: fake_super_add(
                input_ids, request_id, max_new_tokens,
            )
        )
        try:
            class FakeMediator:
                pass
            med = FakeMediator()
            rid = mgr.add_request(
                input_ids=[1, 2, 3],
                request_id="rid_1",
                max_new_tokens=5,
                mediator=med,
                trace_id="trace_xyz",
                saved_names=["logits"],
                expected_count=1,
            )
            assert rid == "rid_1"
            assert "rid_1" in mgr._pending_nnsight_data
            m, tid, names, count = mgr._pending_nnsight_data["rid_1"]
            assert m is med
            assert tid == "trace_xyz"
            assert names == ["logits"]
            assert count == 1
        finally:
            mgr_mod.ContinuousBatchingManager.add_request = original

    def test_add_request_requires_trace_id_when_mediator_given(self):
        from nnsight.modeling.hf_serve.manager import NNsightCBManager

        mgr = NNsightCBManager.__new__(NNsightCBManager)
        mgr._pending_nnsight_data = {}

        import nnsight.modeling.hf_serve.manager as mgr_mod
        original = mgr_mod.ContinuousBatchingManager.add_request
        mgr_mod.ContinuousBatchingManager.add_request = (
            lambda self, input_ids, request_id=None, max_new_tokens=None: "rid_2"
        )
        try:
            class FakeMediator:
                pass
            with pytest.raises(ValueError, match="trace_id"):
                mgr.add_request(
                    input_ids=[1],
                    mediator=FakeMediator(),
                    # trace_id missing
                )
        finally:
            mgr_mod.ContinuousBatchingManager.add_request = original

    def test_add_request_without_mediator_is_passthrough(self):
        """Requests without a mediator don't touch _pending_nnsight_data."""
        from nnsight.modeling.hf_serve.manager import NNsightCBManager

        mgr = NNsightCBManager.__new__(NNsightCBManager)
        mgr._pending_nnsight_data = {}

        import nnsight.modeling.hf_serve.manager as mgr_mod
        original = mgr_mod.ContinuousBatchingManager.add_request
        mgr_mod.ContinuousBatchingManager.add_request = (
            lambda self, input_ids, request_id=None, max_new_tokens=None: "rid_plain"
        )
        try:
            rid = mgr.add_request(input_ids=[1, 2])
            assert rid == "rid_plain"
            assert mgr._pending_nnsight_data == {}
        finally:
            mgr_mod.ContinuousBatchingManager.add_request = original

    def test_register_pending_mediators_forwards_to_helper(self):
        """`_register_pending_mediators` drains stashed data and calls
        `process_new_reqs_direct` with the right tuple shape.
        """
        from nnsight.modeling.hf_serve.manager import NNsightCBManager
        from nnsight.modeling.common.request_helper import NNsightRequestHelper

        mgr = NNsightCBManager.__new__(NNsightCBManager)

        class FakeMediator:
            pass
        med_a, med_b = FakeMediator(), FakeMediator()
        mgr._pending_nnsight_data = {
            "req_a": (med_a, "trace_a", ["x"], 1),
            "req_b": (med_b, "trace_b", [], 2),
            "req_c": (FakeMediator(), "trace_c", [], 1),  # not in this batch
        }

        # Use a real helper and intercept process_new_reqs_direct
        mgr.request_helper = NNsightRequestHelper()
        captured = {"entries": None, "model": None}

        def fake_direct(entries, model):
            captured["entries"] = list(entries)
            captured["model"] = model
        mgr.request_helper.process_new_reqs_direct = fake_direct

        class FakeModel:
            pass
        mgr.nnsight_model = FakeModel()

        # Only req_a and req_b are scheduled this step
        mgr._register_pending_mediators(["req_a", "req_b"])

        # Drained the two scheduled entries, left req_c alone
        assert "req_a" not in mgr._pending_nnsight_data
        assert "req_b" not in mgr._pending_nnsight_data
        assert "req_c" in mgr._pending_nnsight_data

        assert captured["model"] is mgr.nnsight_model
        assert len(captured["entries"]) == 2
        # Entry shape: (req_id, mediator, trace_id, saved_names, expected_count)
        entries_by_id = {e[0]: e for e in captured["entries"]}
        assert entries_by_id["req_a"] == ("req_a", med_a, "trace_a", ["x"], 1)
        assert entries_by_id["req_b"] == ("req_b", med_b, "trace_b", [], 2)

    def test_register_pending_mediators_is_noop_when_empty(self):
        """No pending data → no helper call."""
        from nnsight.modeling.hf_serve.manager import NNsightCBManager
        from nnsight.modeling.common.request_helper import NNsightRequestHelper

        mgr = NNsightCBManager.__new__(NNsightCBManager)
        mgr._pending_nnsight_data = {}
        mgr.request_helper = NNsightRequestHelper()

        called = [False]
        def fake_direct(entries, model):
            called[0] = True
        mgr.request_helper.process_new_reqs_direct = fake_direct
        mgr.nnsight_model = object()

        mgr._register_pending_mediators(["req_x", "req_y"])
        assert called[0] is False


# ------------------------------------------------------------------
# Request helper tests
# ------------------------------------------------------------------

class TestRequestHelper:
    """Test shared NNsightRequestHelper."""

    def test_import(self):
        from nnsight.modeling.common.request_helper import NNsightRequestHelper
        h = NNsightRequestHelper()
        assert h.mediators == {}
        assert h.trace_contexts == {}

    def test_process_batch_groups(self):
        from nnsight.modeling.common.request_helper import NNsightRequestHelper

        h = NNsightRequestHelper()

        # Create mock mediators
        class MockMediator:
            batch_group = None
        class MockInterleaver:
            mediators = []
            class batcher:
                last_batch_group = None
                needs_batching = False
        class MockModel:
            _interleaver = MockInterleaver()

        m1 = MockMediator()
        m2 = MockMediator()
        h.mediators = {"req_0": m1, "req_1": m2}

        num_tokens = {"req_0": 5, "req_1": 3}
        batch_ids = ["req_0", "req_1"]

        h.process_batch_groups(num_tokens, batch_ids, MockModel())

        assert m1.batch_group == [0, 5]
        assert m2.batch_group == [5, 3]
        assert MockModel._interleaver.batcher.last_batch_group == [5, 3]

    def test_match_req_ids_no_suffix(self):
        from nnsight.modeling.common.request_helper import NNsightRequestHelper

        h = NNsightRequestHelper()

        class MockMed:
            pass

        h.mediators = {"req_0": MockMed(), "req_1": MockMed()}

        matched = h.match_req_ids({"req_0", "req_1"}, strip_suffix=False)
        assert len(matched) == 2
        ids = {m[0] for m in matched}
        assert ids == {"req_0", "req_1"}

    def test_match_req_ids_with_suffix(self):
        from nnsight.modeling.common.request_helper import NNsightRequestHelper

        h = NNsightRequestHelper()

        class MockMed:
            pass

        h.mediators = {"req_0-abc123": MockMed()}

        matched = h.match_req_ids({"req_0"}, strip_suffix=True)
        assert len(matched) == 1
        assert matched[0][0] == "req_0"


# ------------------------------------------------------------------
# LanguageModel logits/samples hook points
# ------------------------------------------------------------------

class TestLanguageModelHookPoints:
    """Test that LanguageModel has logits and samples hook points."""

    def test_logits_exists(self, model):
        assert hasattr(model, 'logits')
        assert model.logits is not None

    def test_samples_exists(self, model):
        assert hasattr(model, 'samples')
        assert model.samples is not None


# ------------------------------------------------------------------
# Cross-request batching
# ------------------------------------------------------------------

class TestCrossRequestBatching:
    """Prove that concurrent submissions end up in the same forward pass.

    These tests drive the scheduler directly (not through FastAPI) to
    avoid needing an HTTP server + GPU. We submit requests from two
    concurrent async tasks and assert that at least one ``_step()`` call
    processed tokens from both.
    """

    def test_concurrent_submit_batches_together(self, model):
        """Two concurrent submit_async calls should share forward passes.

        Instruments ``VanillaBatchServer._step`` with a batch-size tracker.
        Submits requests from two concurrent asyncio tasks; after both
        complete, verifies the scheduler saw them together at least once.
        """
        import asyncio
        from transformers import DynamicCache
        from nnsight.modeling.hf_serve.vanilla_server import (
            VanillaBatchServer, VanillaRequest, ActiveRequest,
        )

        server = VanillaBatchServer(model, token_budget=256, max_batch_size=8)

        # Instrument _step to track max batch size seen
        max_batch_seen = {"n": 0}
        original_step = server._step

        def tracking_step(scheduled):
            max_batch_seen["n"] = max(max_batch_seen["n"], len(scheduled))
            return original_step(scheduled)

        server._step = tracking_step

        # Bypass mediator registration (no real nnsight tracing)
        def mock_activate(vr):
            eos_id = getattr(model._model.config, "eos_token_id", None)
            if isinstance(eos_id, list):
                eos_id = eos_id[0]
            active = ActiveRequest(
                req_id=vr.req_id, prompt_ids=vr.token_ids,
                generated_ids=[], max_new_tokens=vr.gen_kwargs.get("max_new_tokens", 5),
                eos_token_id=eos_id or -1,
                past_key_values=DynamicCache(),
                prefilled_len=0, cache_mask=[],
            )
            server._active[vr.req_id] = active
            return active
        server._activate_request = mock_activate

        server.start()

        async def driver():
            async def submit_and_wait(req_id, prompt):
                tokens = model.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
                req = VanillaRequest(
                    req_id=req_id, token_ids=tokens,
                    gen_kwargs={"max_new_tokens": 5}, mediator=None,
                    trace_id=f"trace_{req_id}", saved_names=[], expected_count=1,
                )
                return await server.submit_async(req)

            task_a = asyncio.create_task(submit_and_wait("req_a", "The Eiffel Tower is in"))
            task_b = asyncio.create_task(submit_and_wait("req_b", "Hello world"))
            return await asyncio.gather(task_a, task_b)

        try:
            result_a, result_b = asyncio.run(driver())

            assert result_a is not None
            assert result_b is not None
            assert max_batch_seen["n"] >= 2, (
                f"Expected batch size >= 2 in at least one step, "
                f"saw max {max_batch_seen['n']}"
            )
        finally:
            server.stop()

    def test_submit_async_returns_future(self, model):
        """submit_async returns an awaitable asyncio.Future."""
        import asyncio
        from nnsight.modeling.hf_serve.vanilla_server import (
            VanillaBatchServer, VanillaRequest,
        )

        server = VanillaBatchServer(model)

        async def check():
            req = VanillaRequest(
                req_id="test_future", token_ids=[1, 2, 3],
                gen_kwargs={"max_new_tokens": 1}, mediator=None,
                trace_id="t", saved_names=[], expected_count=1,
            )
            future = server.submit_async(req)
            assert isinstance(future, asyncio.Future)
            assert future.get_loop() is asyncio.get_running_loop()

        asyncio.run(check())

    def test_submit_sync_returns_event(self, model):
        """Sync submit() still returns a threading.Event."""
        import threading
        from nnsight.modeling.hf_serve.vanilla_server import (
            VanillaBatchServer, VanillaRequest,
        )

        server = VanillaBatchServer(model)

        req = VanillaRequest(
            req_id="test_event", token_ids=[1, 2, 3],
            gen_kwargs={"max_new_tokens": 1}, mediator=None,
            trace_id="t", saved_names=[], expected_count=1,
        )
        event = server.submit(req)
        assert isinstance(event, threading.Event)


# ------------------------------------------------------------------
# Mediator timeout
# ------------------------------------------------------------------

class TestMediatorTimeout:
    """Per-mediator timeout prevents a hung intervention from wedging the batch."""

    def test_value_wait_returns_false_on_timeout(self):
        """Value.wait(timeout) returns False when no value arrives in time."""
        import time
        from nnsight.intervention.interleaver import Mediator

        v = Mediator.Value()
        t0 = time.perf_counter()
        result = v.wait(timeout=0.1)
        elapsed = time.perf_counter() - t0
        assert result is False
        assert 0.05 < elapsed < 0.5

    def test_value_wait_returns_true_after_put(self):
        """Value.wait(timeout) returns True when a value arrives before timeout."""
        import threading, time
        from nnsight.intervention.interleaver import Mediator

        v = Mediator.Value()

        def delayed_put():
            time.sleep(0.05)
            v.put("payload")

        threading.Thread(target=delayed_put, daemon=True).start()
        result = v.wait(timeout=1.0)
        assert result is True
        assert v.get() == "payload"

    def test_value_wait_no_timeout_blocks_until_put(self):
        """Without timeout, wait() blocks until a value arrives."""
        import threading, time
        from nnsight.intervention.interleaver import Mediator

        v = Mediator.Value()

        def delayed_put():
            time.sleep(0.1)
            v.put("ok")

        threading.Thread(target=delayed_put, daemon=True).start()
        t0 = time.perf_counter()
        result = v.wait()  # no timeout → blocks
        elapsed = time.perf_counter() - t0
        assert result is True
        assert 0.05 < elapsed < 0.5

    def test_hung_intervention_times_out(self, model):
        """A hung intervention times out with a warning, doesn't hang the trace.

        Sets a short mediator_timeout on the interleaver, runs a trace where
        the intervention sleeps past the timeout, and verifies the trace
        completes (via warning) rather than hanging forever.
        """
        import time
        import warnings as _warnings

        original_timeout = model._interleaver.mediator_timeout
        model._interleaver.mediator_timeout = 0.3

        try:
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                t0 = time.perf_counter()

                # A trace whose intervention hangs for 3s with a 0.3s timeout.
                # Without the timeout mechanism, this test would hang forever.
                with model.trace("Hello"):
                    time.sleep(3.0)
                    _ = model.lm_head.output.save()

                elapsed = time.perf_counter() - t0

            # Should have timed out within ~1s, not waited the full 3s
            assert elapsed < 2.0, (
                f"Expected trace to abort within ~1s via timeout; "
                f"took {elapsed:.2f}s (timeout may not be firing)"
            )
            timeout_warnings = [
                w for w in caught
                if issubclass(w.category, RuntimeWarning)
                and "timed out" in str(w.message)
            ]
            assert timeout_warnings, (
                f"Expected RuntimeWarning about timeout; "
                f"got: {[str(w.message) for w in caught]}"
            )
        finally:
            model._interleaver.mediator_timeout = original_timeout


class TestEndToEndTrace:
    """Drive a real nnsight trace through VanillaBatchServer.

    Catches the class of bugs where mediator workers execute user code
    outside the interleaver context (``_interleaving=False``), which
    surfaces as ``ValueError: The model did not execute`` from
    ``envoy.output``. Previous tests mocked ``_activate_request``, so
    that path was never exercised.
    """

    def _reset_interleaver(self, model):
        """Clear any mediators the server left on the module-scoped model.

        The server path intentionally leaves the interleaver live across
        generation steps, but the ``model`` fixture is shared with tests
        that use the default local trace path — stale mediators from a
        server test would leak into the next local trace.
        """
        iv = model._interleaver
        if iv.mediators:
            for mediator in list(iv.mediators):
                try:
                    mediator.cancel()
                except Exception:
                    pass
            iv.mediators = []

    def _make_server_backend(self, server):
        """A local backend mirroring ``api/server.py``'s submission flow."""
        from nnsight.intervention.backends import Backend
        from nnsight.intervention.tracing.globals import Globals

        class VanillaServerBackend(Backend):
            def __init__(self_inner, srv):
                self_inner.server = srv

            def __call__(self_inner, tracer):
                if tracer is None:
                    return
                interventions = Backend.__call__(self_inner, tracer)
                try:
                    Globals.enter()
                    _args, kwargs = tracer._setup_interleaver(interventions)
                    entries = self_inner.server.build_entries(kwargs)
                    tracer.mediators.clear()
                    pending = [
                        (e, self_inner.server.submit(e)) for e in entries
                    ]
                finally:
                    Globals.exit()
                all_saves = {}
                for entry, event in pending:
                    got = event.wait(timeout=30.0)
                    assert got, f"Timed out waiting on {entry.req_id}"
                    saves = self_inner.server.get_result(entry.req_id)
                    if saves and "__error__" in saves:
                        raise RuntimeError(
                            f"Server reported error for {entry.req_id}: "
                            f"{saves['__error__']}"
                        )
                    if saves:
                        all_saves.update(saves)
                tracer.push(all_saves)

        return VanillaServerBackend(server)

    def test_real_trace_captures_activations(self, model):
        """End-to-end: user code runs, saves come back as real tensors.

        The server generates ``max_new_tokens`` decode steps after
        prefill, so the captured tensor corresponds to whichever forward
        pass satisfied the ``.output`` hook first (prefill's last-layer
        activations). We assert shape + non-zero content, not
        numerical equality against a direct forward — that comparison
        would need to fix the generation step the save targets.
        """
        from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

        server = VanillaBatchServer(model, max_batch_size=2)
        server.start()
        backend = self._make_server_backend(server)

        try:
            with model.trace("Hello world", backend=backend):
                hidden = model.transformer.h[0].output.save()
                logits = model.lm_head.output.save()
        finally:
            server.stop()
            self._reset_interleaver(model)

        # User intervention code actually ran and produced real tensors.
        assert hidden is not None and hidden.dim() == 3
        assert logits is not None and logits.dim() == 3
        assert hidden.shape[-1] == 768
        assert logits.shape[-1] == model._model.config.vocab_size
        # Not all zeros — real activations were captured, not a placeholder.
        assert hidden.abs().max().item() > 0
        assert logits.abs().max().item() > 0

    def test_real_trace_intervention_applied(self, model):
        """Interventions written against the server also mutate activations."""
        from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

        server = VanillaBatchServer(model, max_batch_size=2)
        server.start()
        backend = self._make_server_backend(server)

        try:
            with model.trace("Hello world", backend=backend):
                model.transformer.h[0].output[:] = 0
                post_zero = model.transformer.h[0].output.save()
                final = model.lm_head.output.save()
        finally:
            server.stop()
            self._reset_interleaver(model)

        assert torch.all(post_zero == 0), (
            "In-place zeroing did not take effect — intervention path broken"
        )
        # Logits must DIFFER from the un-patched forward now.
        tokens = model.tokenizer("Hello world", return_tensors="pt")
        input_ids = tokens["input_ids"].to(model.device)
        with torch.no_grad():
            direct = model._model(input_ids=input_ids)
        assert not torch.allclose(final, direct.logits, atol=1e-3)


# ------------------------------------------------------------------
# Multi-GPU: cache merge/split must respect per-layer devices under
# ``device_map``-sharded models. Runs only with ≥2 CUDA GPUs.
# ------------------------------------------------------------------

_HAS_2GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2


@pytest.fixture(scope="module")
def split_model():
    """GPT-2 split across cuda:0 (layers 0-5) and cuda:1 (layers 6-11)."""
    if not _HAS_2GPU:
        pytest.skip(f"needs ≥2 GPUs, have {torch.cuda.device_count()}")
    from nnsight import LanguageModel

    # Tied weights: lm_head shares weight with transformer.wte, so they
    # must co-locate. Put embedding + lm_head on cuda:0.
    device_map = {
        "transformer.wte": 0,
        "transformer.wpe": 0,
        "transformer.drop": 0,
        "transformer.h.0": 0, "transformer.h.1": 0, "transformer.h.2": 0,
        "transformer.h.3": 0, "transformer.h.4": 0, "transformer.h.5": 0,
        "transformer.h.6": 1, "transformer.h.7": 1, "transformer.h.8": 1,
        "transformer.h.9": 1, "transformer.h.10": 1, "transformer.h.11": 1,
        "transformer.ln_f": 1,
        "lm_head": 0,
    }
    m = LanguageModel(
        "openai-community/gpt2",
        device_map=device_map,
        dispatch=True,
    )
    yield m


@pytest.mark.skipif(not _HAS_2GPU, reason="needs ≥2 CUDA GPUs")
class TestMultiGPUCache:
    """Verify _merge_caches / _split_cache work under ``device_map``-split
    models. Guards against the bug where the merged KV cache is
    allocated on a single ``model.device`` — attention on a layer that
    lives elsewhere then crashes with ``Expected all tensors to be on
    the same device`` inside ``torch.cat``.
    """

    def _reset_interleaver(self, model):
        iv = model._interleaver
        if iv.mediators:
            for mediator in list(iv.mediators):
                try:
                    mediator.cancel()
                except Exception:
                    pass
            iv.mediators = []

    def _make_server_backend(self, server):
        """Mirror ``api/server.py`` submission flow without HTTP."""
        from nnsight.intervention.backends import Backend
        from nnsight.intervention.tracing.globals import Globals

        class VanillaServerBackend(Backend):
            def __init__(self_inner, srv):
                self_inner.server = srv

            def __call__(self_inner, tracer):
                if tracer is None:
                    return
                interventions = Backend.__call__(self_inner, tracer)
                try:
                    Globals.enter()
                    _args, kwargs = tracer._setup_interleaver(interventions)
                    entries = self_inner.server.build_entries(kwargs)
                    tracer.mediators.clear()
                    pending = [
                        (e, self_inner.server.submit(e)) for e in entries
                    ]
                finally:
                    Globals.exit()
                all_saves = {}
                for entry, event in pending:
                    got = event.wait(timeout=60.0)
                    assert got, f"Timed out waiting on {entry.req_id}"
                    saves = self_inner.server.get_result(entry.req_id)
                    if saves and "__error__" in saves:
                        raise RuntimeError(
                            f"Server reported error for {entry.req_id}: "
                            f"{saves['__error__']}"
                        )
                    if saves:
                        all_saves.update(saves)
                tracer.push(all_saves)

        return VanillaServerBackend(server)

    def test_layers_actually_split_across_gpus(self, split_model):
        """Sanity: sharding took effect; early and late layers are on
        different CUDA devices."""
        hf = split_model._model
        early_dev = next(hf.transformer.h[2].parameters()).device
        late_dev = next(hf.transformer.h[9].parameters()).device
        assert early_dev.type == "cuda" and early_dev.index == 0
        assert late_dev.type == "cuda" and late_dev.index == 1

    def test_merged_cache_layer_devices(self, split_model):
        """Directly assert ``_merge_caches`` places per-layer K/V on the
        layer's own device.

        Populates per-request caches with tensors on cuda:0 (as if from
        a previous step that hadn't yet been device-corrected), calls
        ``_merge_caches``, and checks that layer 2's merged K/V lives on
        cuda:0 while layer 9's lives on cuda:1 — i.e. the merge code is
        copying per-layer onto the layer's device.
        """
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer
        from nnsight.modeling.hf_serve.vanilla_server import (
            VanillaBatchServer, ActiveRequest, ScheduledItem,
        )

        server = VanillaBatchServer(split_model)
        num_heads = split_model._model.config.n_head
        head_dim = split_model._model.config.n_embd // num_heads
        num_layers = split_model._model.config.n_layer
        seq_len = 4

        def make_cache_on(device):
            cache = DynamicCache()
            for _ in range(num_layers):
                k = torch.zeros(1, num_heads, seq_len, head_dim, device=device)
                v = torch.zeros_like(k)
                layer = DynamicLayer()
                layer.update(k, v)
                cache.layers.append(layer)
            return cache

        reqs = []
        for i in range(2):
            reqs.append(ActiveRequest(
                req_id=f"req_{i}",
                prompt_ids=[0] * seq_len,
                generated_ids=[],
                max_new_tokens=1,
                eos_token_id=-1,
                past_key_values=make_cache_on(torch.device("cuda:0")),
                prefilled_len=seq_len,
                cache_mask=[1] * seq_len,
            ))
        scheduled = [
            ScheduledItem(request=r, num_tokens=1, is_prefill=False, token_ids=[0])
            for r in reqs
        ]

        merged = server._merge_caches(scheduled, max_cache_len=seq_len)
        assert merged is not None
        assert len(merged.layers) == num_layers
        assert merged.layers[2].keys.device.index == 0
        assert merged.layers[9].keys.device.index == 1
        assert merged.layers[2].values.device.index == 0
        assert merged.layers[9].values.device.index == 1

    def test_prefill_decode_cycle_on_split_model(self, split_model):
        """End-to-end trace over a device_map-split model.

        Exercises ``_merge_caches`` / ``_split_cache`` on every decode
        step. If those allocated the merged cache on a single device,
        attention on layer 9 (cuda:1) would crash with a device-mismatch
        error inside ``torch.cat``.
        """
        from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

        server = VanillaBatchServer(split_model, max_batch_size=2)
        server.start()
        backend = self._make_server_backend(server)

        try:
            with split_model.trace("Hello world", backend=backend):
                early = split_model.transformer.h[2].output.save()
                late = split_model.transformer.h[9].output.save()
        finally:
            server.stop()
            self._reset_interleaver(split_model)

        assert early is not None and early.dim() == 3
        assert late is not None and late.dim() == 3
        assert early.shape[-1] == 768
        assert late.shape[-1] == 768
        # Saved tensors come back on their originating layer's device.
        assert early.device.type == "cuda" and early.device.index == 0, (
            f"early (layer 2) expected cuda:0, got {early.device}"
        )
        assert late.device.type == "cuda" and late.device.index == 1, (
            f"late (layer 9) expected cuda:1, got {late.device}"
        )
        assert early.abs().max().item() > 0
        assert late.abs().max().item() > 0

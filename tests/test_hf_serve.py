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
        with model.trace("Hello"):
            hs = model.transformer.h[0].output[0].save()
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

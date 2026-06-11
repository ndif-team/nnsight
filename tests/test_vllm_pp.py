import inspect
import threading
import time
from collections import defaultdict, namedtuple
from unittest.mock import MagicMock, patch

import pytest
import torch

from nnsight.modeling.vllm.lazy_remote_tensor import (
    NOT_ON_THIS_RANK,
    LazyRemoteTensor,
    merge_saved,
    strip_lazy,
)
from nnsight.modeling.vllm.pp_listener import PPListener
from vllm.model_executor.models.utils import PPMissingLayer


class TestLazyRemoteTensor:

    def _make_lazy(self, real_tensor=None):
        """Helper: create a LazyRemoteTensor with optional pre-set real tensor."""
        lazy = LazyRemoteTensor(
            source_rank=1,
            provider_string="model.layers.50.output.i0",
            dtype=torch.float32,
        )
        if real_tensor is not None:
            lazy._real = real_tensor
        return lazy

    def test_metadata_no_materialization(self):
        lazy = self._make_lazy()
        assert lazy.dtype == torch.float32
        assert lazy._real is None

    def test_setitem_noop(self):
        lazy = self._make_lazy()
        lazy[:] = torch.zeros(1, 5, 768)
        assert lazy._real is None  # no materialization

    def test_getitem_returns_child_lazy(self):
        lazy = self._make_lazy()
        result = lazy[0]
        assert isinstance(result, LazyRemoteTensor)
        assert result is not lazy  # child with deferred indexing

    def test_save_returns_self(self):
        lazy = self._make_lazy()
        result = lazy.save()
        assert result is lazy
        assert lazy._real is None

    def test_torch_function_materializes(self):
        real = torch.randn(1, 5, 768)
        lazy = self._make_lazy(real_tensor=real)
        result = lazy + 1
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, real + 1)

    def test_torch_function_in_args(self):
        real = torch.randn(1, 5, 768)
        lazy = self._make_lazy(real_tensor=real)
        result = torch.sum(lazy)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.sum(real))

    def test_comparison_ops_use_real_value(self):
        """``lazy == x`` (and the other comparisons) must compare against the
        materialized value, like the arithmetic dunders do.

        Without explicit comparison dunders Python falls back to identity for
        ``==``/``!=`` (returning a plain ``False``/``True`` bool instead of an
        elementwise tensor) and raises for the orderings — on the non-owning
        rank only, so user code branching on a comparison silently diverges
        between ranks instead of erroring.
        """
        real = torch.tensor([1.0, 2.0, 3.0])
        lazy = self._make_lazy(real_tensor=real)

        eq = lazy == torch.tensor([1.0, 0.0, 3.0])
        assert isinstance(eq, torch.Tensor), f"== fell back to identity: {eq!r}"
        assert eq.tolist() == [True, False, True]

        ne = lazy != torch.tensor([1.0, 0.0, 3.0])
        assert isinstance(ne, torch.Tensor)
        assert ne.tolist() == [False, True, False]

        lt = lazy < 2.5
        assert isinstance(lt, torch.Tensor)
        assert lt.tolist() == [True, True, False]

        ge = lazy >= 2.0
        assert isinstance(ge, torch.Tensor)
        assert ge.tolist() == [False, True, True]

    def test_hashable_with_comparison_ops(self):
        """Defining ``__eq__`` removes the default ``__hash__``; the lazy must
        stay hashable (identity hash) so it can live in sets/dict keys."""
        lazy = self._make_lazy()
        s = {lazy}
        assert lazy in s

    def test_unmaterialized_comparison_does_not_pull(self):
        """A comparison on an unmaterialized lazy with no pull function should
        raise the standard materialize error — not silently return identity."""
        lazy = self._make_lazy()
        with pytest.raises(RuntimeError, match="no pull function"):
            _ = lazy == torch.zeros(3)


class TestStripLazyContainers:
    """Collection-path container handling (``strip_lazy`` / ``merge_saved``)."""

    Point = namedtuple("Point", ["hidden", "residual"])

    def _lazy(self):
        return LazyRemoteTensor(
            source_rank=1, provider_string="m.x.output.i0", dtype=torch.float32
        )

    def test_strip_lazy_namedtuple(self):
        """A NamedTuple save containing a lazy must round-trip: NamedTuple
        constructors take positional fields, so ``type(value)(generator)``
        raises TypeError and the whole collection pass dies."""
        value = self.Point(self._lazy(), torch.ones(2))
        stripped, has_real, has_lazy = strip_lazy(value)
        assert isinstance(stripped, self.Point)
        assert stripped.hidden is NOT_ON_THIS_RANK
        assert torch.equal(stripped.residual, torch.ones(2))
        assert has_real and has_lazy

    def test_merge_saved_preserves_namedtuple_type(self):
        a = self.Point(NOT_ON_THIS_RANK, torch.ones(2))
        b = self.Point(torch.zeros(2), NOT_ON_THIS_RANK)
        merged = merge_saved(a, b)
        assert isinstance(merged, self.Point)
        assert torch.equal(merged.hidden, torch.zeros(2))
        assert torch.equal(merged.residual, torch.ones(2))


class TestProviderModulePath:
    """``_provider_to_module_path`` must keep root wrapper-module names.

    For sub-envoy providers the suffix is ``.output.iN``/``.input.iN`` and both
    trailing parts strip. But a root eproperty's provider is
    ``model.logits.iN`` — stripping two parts collapses BOTH ``model.logits``
    and ``model.samples`` to ``"model"``, so their meta/shape-cache entries
    collide (latent: masked today only because consumers always request the
    shape-on-wire mode).
    """

    def test_strips_output_and_input_suffix(self):
        from nnsight.modeling.vllm.pp_listener import _provider_to_module_path

        assert _provider_to_module_path("model.layers.5.output.i0") == "model.layers.5"
        assert _provider_to_module_path("model.layers.5.input.i12") == "model.layers.5"

    def test_keeps_root_wrapper_module_name(self):
        from nnsight.modeling.vllm.pp_listener import _provider_to_module_path

        assert _provider_to_module_path("model.logits.i0") == "model.logits"
        assert _provider_to_module_path("model.samples.i3") == "model.samples"


class TestPPListener:

    def _make_listener(self, buffer=None):
        """Helper: create a PPListener for local_lookup tests (no pull_group needed)."""
        if buffer is None:
            buffer = {}
        cond = threading.Condition()
        return PPListener(
            buffer=buffer,
            condition=cond,
            pull_group=None,
            local_rank=0,
            device=torch.device("cpu"),
        ), cond

    def test_serve_existing_value(self):
        """Listener serves a value that's already in the buffer."""
        buf = {"model.layers.5.output.i0": torch.randn(1, 5, 768)}
        listener, cond = self._make_listener(buf)

        result = listener.local_lookup("model.layers.5.output.i0")
        assert torch.equal(result, buf["model.layers.5.output.i0"])

    def test_wait_for_value(self):
        """Listener waits until a value appears in the buffer."""
        listener, cond = self._make_listener()

        result_holder = [None]

        def lookup():
            result_holder[0] = listener.local_lookup(
                "model.layers.5.output.i0", timeout=5.0
            )

        t = threading.Thread(target=lookup)
        t.start()

        # Value not yet in buffer — thread is waiting
        time.sleep(0.05)
        assert result_holder[0] is None

        # Add value and notify
        tensor = torch.randn(1, 5, 768)
        with cond:
            listener._buffer["model.layers.5.output.i0"] = tensor
            cond.notify_all()

        t.join(timeout=5.0)
        assert result_holder[0] is not None
        assert torch.equal(result_holder[0], tensor)

    def test_timeout_raises(self):
        """Listener raises TimeoutError if value never appears."""
        listener, cond = self._make_listener()

        with pytest.raises(TimeoutError):
            listener.local_lookup("missing.key", timeout=0.1)

    def test_concurrent_pulls_routed_by_tag(self):
        """Many consumer threads pulling at once each get exactly their own
        reply — routed by the per-pull response tag, with no lock.

        gloo routes point-to-point traffic by ``(peer, tag)``, never by content,
        so concurrent consumers can only be disambiguated by giving each pull its
        own response tag (carried in the request). This drives N real threads
        through ``pull_from_remote`` with a mocked ``dist`` standing in for a
        producer that serves replies **out of arrival order**: the request send
        records the per-pull tag → expected value, and the response recv fills
        the buffer with the value for *that tag*. The test asserts every caller
        gets its own value back. If responses ever shared one tag (the old bug),
        the mock could not tell them apart and the values would cross.
        """
        N = 12
        prov = "decoder.blocks.7.output.i0"          # non-standard naming on purpose
        mod = "decoder.blocks.7"
        # Precomputed-path metadata so each response is a single flat recv.
        meta_map = {mod: {"dtype": torch.float32, "module_shapes": [(1,)], "num_outputs": 1}}
        listener = PPListener(
            buffer={}, condition=threading.Condition(), pull_group=MagicMock(),
            local_rank=0, device=torch.device("cpu"), meta_map=meta_map,
        )

        import nnsight.modeling.vllm.pp_listener as L

        pending = {}                     # response_tag -> value the producer will return
        request_tags = []                # every tag seen on a request send
        lock = threading.Lock()

        def fake_send(tensor, group=None, group_dst=None, tag=None):
            if tag == L.TAG_REQUEST:
                _rank, rtag, _hnt, key = L._decode_request(tensor)
                idx = int(key.split("|", 1)[0][3:])     # "req<idx>" -> idx
                with lock:
                    pending[rtag] = float(idx)
                    request_tags.append(rtag)

        def fake_recv(tensor, group=None, group_src=None, tag=None):
            # Precomputed path: one flat recv on the pull's private tag.
            with lock:
                val = pending.get(tag)
            assert val is not None, f"recv on tag {tag!r} with no matching request"
            tensor.fill_(val)

        fake_dist = MagicMock()
        fake_dist.send.side_effect = fake_send
        fake_dist.recv.side_effect = fake_recv

        got = {}

        def pull(i):
            out = listener.pull_from_remote(
                source_rank=1, provider_string=prov, num_tokens=1, req_id=f"req{i}",
            )
            got[i] = float(out.reshape(-1)[0].item())

        with patch.object(L, "dist", fake_dist):
            threads = [threading.Thread(target=pull, args=(i,)) for i in range(N)]
            for t in threads: t.start()
            for t in threads: t.join(timeout=5.0)
            assert all(not t.is_alive() for t in threads)

        # Every caller got ITS OWN value — no cross-delivery.
        assert got == {i: float(i) for i in range(N)}, got
        # Each pull used a distinct response tag, all ≥ base, none aliasing TAG_REQUEST.
        assert len(set(request_tags)) == N, request_tags
        assert all(t >= L.TAG_RESPONSE_BASE for t in request_tags)


class TestEnvoyPPMissingShortCircuit:

    def _make_pp_envoy(self):
        """Create a minimal PPEnvoy setup to test PPMissing short-circuit."""
        from nnsight.intervention.interleaver import Interleaver
        from nnsight.modeling.vllm.pp_envoy import PPEnvoy

        module = PPMissingLayer()
        interleaver = Interleaver()
        interleaver.pp_enabled = True
        interleaver.pp_local_rank = 0
        interleaver.pp_listener = None

        mock_map = MagicMock()
        mock_map.get_owning_rank.return_value = 1
        mock_map.is_local.return_value = False
        interleaver.pp_module_map = mock_map
        interleaver.pp_module_meta = {"model.layers.50": {"dtype": torch.float32}}

        mediator = MagicMock()
        mediator.iteration_tracker = defaultdict(int)
        mediator.iteration = None
        mediator.batch_group = None
        mediator.pp_req_id = None
        interleaver.current = mediator
        interleaver._interleaving = True
        interleaver.mediators = [mediator]

        envoy = PPEnvoy(module, path="model.layers.50", interleaver=interleaver)
        return envoy, mediator

    def test_output_returns_lazy_tensor(self):
        envoy, mediator = self._make_pp_envoy()
        result = envoy.output
        assert isinstance(result, LazyRemoteTensor)
        assert result._meta["provider_string"] == "model.layers.50.output.i0"
        assert result._meta["source_rank"] == 1

    def test_output_increments_tracker(self):
        envoy, mediator = self._make_pp_envoy()
        assert mediator.iteration_tracker["model.layers.50.output"] == 0
        _ = envoy.output
        assert mediator.iteration_tracker["model.layers.50.output"] == 1
        _ = envoy.output
        assert mediator.iteration_tracker["model.layers.50.output"] == 2

    def test_output_setter_noop_for_pp_missing(self):
        envoy, mediator = self._make_pp_envoy()
        # Should not raise or block
        envoy.output = torch.zeros(1)

    def test_unknown_owner_consume_raises_not_hangs(self):
        """A stub module whose owner the ownership map can't resolve (e.g. a
        non-standard module name like Falcon's ``word_embeddings``) builds a
        lazy with ``source_rank=None``. Consuming it must raise a descriptive
        error — today the misdirected pull surfaces as a distributed hang or
        crash. A never-consumed lazy must stay harmless (saves merge away)."""
        envoy, mediator = self._make_pp_envoy()
        interleaver = envoy.interleaver
        interleaver.pp_module_map.get_owning_rank.return_value = None
        interleaver.pp_listener = MagicMock()  # wired but must never be hit

        lazy = envoy.output          # build is fine (and save() would be too)
        assert isinstance(lazy, LazyRemoteTensor)

        with pytest.raises(RuntimeError, match="owning .*rank|owner"):
            _ = lazy + 1

        interleaver.pp_listener.pull_from_remote.assert_not_called()

    def test_root_eproperty_dtype_meta_resolved_from_full_key(self):
        """The dtype hint for a root eproperty (``logits``/``samples``) must be
        looked up under the full ``{path}.{key}`` — the same lookup the
        source-rank resolution uses — not ``path`` alone (which is just
        ``"model"`` for root epropertys, so the lookup misses and the lazy
        carries the ``float32`` fallback)."""
        from nnsight.modeling.vllm.pp_envoy import _pp_lazy_access

        envoy, mediator = self._make_pp_envoy()
        interleaver = envoy.interleaver
        interleaver.pp_module_meta = {"model.samples": {"dtype": torch.int32}}

        class RootHost:
            path = "model"

        host = RootHost()
        host.interleaver = interleaver

        lazy = _pp_lazy_access(host, "samples")
        assert lazy._meta["dtype"] == torch.int32


class TestIteratorMediatorResolution:
    """The iterator must resolve its mediator from the worker's thread-local,
    not the shared ``interleaver.current`` slot.

    ``current`` is only valid under strict lockstep: it is reassigned by every
    ``Mediator.start()`` and around every ``handle``. Under PP a worker runs
    concurrently with both (run-ahead after a leading cross-stage RELEASE), so
    a worker entering its iter loop can observe ANOTHER invoke's mediator (or
    None) in ``current`` and corrupt its iteration state. ``eproperty.__get__``
    and ``_pp_lazy_access`` already resolve via ``current_mediator()``; the
    iterator is the remaining bare read.
    """

    def test_iter_uses_thread_local_mediator(self):
        from nnsight.intervention.interleaver import _active_mediator
        from nnsight.intervention.tracing.iterator import IteratorTracer

        my_mediator = MagicMock(name="my_mediator")
        other_mediator = MagicMock(name="other_mediator")
        my_mediator._pp_worker_iteration = "untouched"
        other_mediator._pp_worker_iteration = "untouched"

        interleaver = MagicMock()
        # The shared slot points at ANOTHER invoke's mediator — exactly the
        # run-ahead race (start()/handle reassigned it after this worker
        # released the forward).
        interleaver.current = other_mediator

        model = MagicMock()
        model.modules.return_value = []  # no hooks to register

        tracer = IteratorTracer.__new__(IteratorTracer)
        tracer.interleaver = interleaver
        tracer.iteration = slice(0, 1)
        tracer.model = model

        _active_mediator.value = my_mediator
        try:
            gen = iter(tracer)
            next(gen)
            # The worker's OWN mediator must carry the iteration bookkeeping...
            assert my_mediator._pp_worker_iteration == 0, (
                "iterator bound to interleaver.current instead of the "
                "worker thread-local mediator"
            )
            # ...and the other invoke's mediator must be untouched.
            assert other_mediator._pp_worker_iteration == "untouched"
            gen.close()
        finally:
            _active_mediator.value = None

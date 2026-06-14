import inspect
import os
import threading
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


class TestRankDivergenceTripwire:
    """A real/real merge clash where the two ranks' copies genuinely DIFFER
    must warn (``PPRankDivergenceWarning``) — the silent later-rank-wins pick
    is how in-trace randomness produced plausible-but-wrong results. Identical
    redundant copies (the normal case: every rank computed the value
    deterministically from the same inputs) must merge silently.
    """

    def _no_warning(self, a, b):
        import warnings as w

        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            merged = merge_saved(a, b, label="slot")
        assert not [
            c for c in caught if issubclass(c.category, PPRankDivergenceWarning)
        ], [str(c.message) for c in caught]
        return merged

    def test_identical_tensors_merge_silently(self):
        t = torch.arange(6.0).reshape(2, 3)
        merged = self._no_warning(t, t.clone())
        assert torch.equal(merged, t)

    def test_low_order_float_noise_is_tolerated(self):
        # Redundant cross-rank execution of the same math may differ in the
        # last bits; that is not divergence.
        t = torch.randn(4, 8, dtype=torch.float32)
        self._no_warning(t, t + 1e-8)

    def test_sentinel_real_merge_silently(self):
        t = torch.ones(3)
        merged = self._no_warning(NOT_ON_THIS_RANK, t)
        assert torch.equal(merged, t)

    def test_divergent_tensors_warn_with_label(self):
        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        a, b = torch.zeros(5), torch.ones(5)
        with pytest.warns(PPRankDivergenceWarning, match="'noise'"):
            merged = merge_saved(a, b, label="noise")
        assert torch.equal(merged, b)  # later rank still wins (same degrade)

    def test_divergent_scalars_warn(self):
        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        with pytest.warns(PPRankDivergenceWarning, match="3 vs 7"):
            assert merge_saved(3, 7, label="count") == 7

    def test_nested_label_names_the_slot(self):
        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        a = {"steps": [torch.zeros(2), torch.zeros(2)]}
        b = {"steps": [torch.zeros(2), torch.ones(2)]}
        with pytest.warns(PPRankDivergenceWarning, match=r"outs\['steps'\]\[1\]"):
            merge_saved(a, b, label="outs")

    def test_shape_mismatch_warns(self):
        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        with pytest.warns(PPRankDivergenceWarning, match="shape"):
            merge_saved(torch.zeros(2, 3), torch.zeros(3, 2), label="h")

    def test_exception_envelope_bypasses_tripwire(self):
        # Both ranks defer the same user error with per-rank tracebacks
        # (device names differ) — must union silently, not warn.
        import pickle
        import warnings as w

        import zstandard

        from nnsight.modeling.vllm.collect import merge_collected_saves
        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        def pack(obj):
            return zstandard.ZstdCompressor(level=1).compress(pickle.dumps(obj))

        r0 = pack({"req": {"__nnsight_exceptions__": {"req": ("E", "tb cuda:0")}}})
        r1 = pack({"req": {"__nnsight_exceptions__": {"req": ("E", "tb cuda:1")}}})
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            merged = merge_collected_saves([r0, r1])
        assert not [
            c for c in caught if issubclass(c.category, PPRankDivergenceWarning)
        ]
        assert merged["req"]["__nnsight_exceptions__"]["req"][1] == "tb cuda:1"


class TestMergeUnion:
    """The merge is a positional union over the sentinel (= stage-ownership)
    encoding: prefer real over sentinel at every position, recurse where both
    are real, length-tolerant for lists. Each test maps to one way two ranks'
    save trees can legitimately differ in structure.
    """

    NOR = NOT_ON_THIS_RANK

    def _merge_quiet(self, a, b, label="outs"):
        import warnings as w

        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            out = merge_saved(a, b, label=label)
        warned = [
            c for c in caught if issubclass(c.category, PPRankDivergenceWarning)
        ]
        return out, warned

    def test_owner_real_beats_nonowner_sentinels_equal_length(self):
        # The common per-step append of a single-stage module value: the owner
        # has reals, every other rank has same-length all-sentinel lists.
        owner = [torch.tensor([float(i)]) for i in range(4)]
        other = [self.NOR] * 4
        out, warned = self._merge_quiet(other, owner)  # non-owner is 'a'
        assert not warned
        assert [t.item() for t in out] == [0.0, 1.0, 2.0, 3.0]

    def test_overshoot_sentinel_tail_dropped_silently(self):
        # Run-ahead: one rank looped one step past generation end and appended
        # an unconsumed lazy (-> sentinel). Effective lengths match -> silent;
        # the overshoot tail is dropped.
        a = [torch.zeros(2), torch.ones(2), self.NOR]   # 2 real + overshoot
        b = [torch.zeros(2), torch.ones(2)]
        out, warned = self._merge_quiet(a, b)
        assert not warned
        assert len(out) == 2 and torch.equal(out[1], torch.ones(2))

    def test_stalled_worker_shorter_real_list_warns_but_keeps_complete(self):
        # Mechanism 1/3: a worker stalled (or errored) before the last step, so
        # its list is genuinely one real entry short. The union keeps the
        # complete side AND announces the gap.
        complete = [torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0])]
        short = [torch.tensor([0.0]), torch.tensor([1.0])]
        out, warned = self._merge_quiet(short, complete)
        assert len(warned) == 1
        assert "never reached" in str(warned[0].message)
        assert [t.item() for t in out] == [0.0, 1.0, 2.0]

    def test_empty_vs_populated_keeps_populated(self):
        # The historical clobber: one rank's list stayed empty. Populated wins
        # (data not lost), and the asymmetry is announced.
        out, warned = self._merge_quiet([], [torch.ones(2), torch.ones(2)])
        assert len(out) == 2
        assert len(warned) == 1

    def test_mixed_stage_list_unions_per_position(self):
        # Each step appends a cross-stage pair: stage A owns even indices,
        # stage B owns odd (sentinel where not owned). Union reconstructs the
        # full interleaved list with no warning.
        a = [torch.tensor([0.0]), self.NOR, torch.tensor([2.0]), self.NOR]
        b = [self.NOR, torch.tensor([1.0]), self.NOR, torch.tensor([3.0])]
        out, warned = self._merge_quiet(a, b)
        assert not warned
        assert [t.item() for t in out] == [0.0, 1.0, 2.0, 3.0]

    def test_nested_list_in_dict_unions(self):
        # tracer.cache-style: per-stage dicts with disjoint keys, each value a
        # per-step list. Union by key, union each list, no warning.
        a = {"s0": [torch.zeros(1), torch.zeros(1)]}
        b = {"s1": [torch.ones(1), torch.ones(1)]}
        out, warned = self._merge_quiet(a, b)
        assert not warned
        assert set(out) == {"s0", "s1"} and len(out["s0"]) == 2

    def test_type_clash_warns_and_falls_through_to_leaf_degrade(self):
        # The one non-union path: incompatible structures at one slot. No
        # positional union applies, so it falls through to the leaf degrade —
        # _divergence_detail describes the mismatch, the warning fires, b wins.
        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        a = torch.zeros(2)                       # tensor
        b = [torch.zeros(2), torch.zeros(2)]     # list
        with pytest.warns(PPRankDivergenceWarning, match="mismatch"):
            kept = merge_saved(a, b, label="x")
        assert kept is b

    def test_equal_length_divergent_prefix_still_warns(self):
        # Length-tolerant union must not mask a value divergence on a shared
        # position — the per-element tripwire still fires.
        from nnsight.modeling.vllm.lazy_remote_tensor import (
            PPRankDivergenceWarning,
        )

        a = [torch.zeros(2), torch.zeros(2)]
        b = [torch.zeros(2), torch.ones(2)]
        with pytest.warns(PPRankDivergenceWarning, match=r"outs\[1\]"):
            merge_saved(a, b, label="outs")


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
        """Helper: create a PPListener with no pull group (in-process tests)."""
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

    def test_dispatch_parked_serves_waiting_request(self):
        """A request parked for a not-yet-produced value is handed to the
        reply pool — with that value — the moment the producer writes it."""
        listener, cond = self._make_listener()
        served = []
        listener._serve_reply = lambda req, value: served.append((req, value))

        key = ("model.layers.5.output.i0", "reqA")
        parked_req = (1, 1024)
        with cond:
            listener._parked[key] = [parked_req]

        tensor = torch.randn(1, 5, 768)
        listener.dispatch_parked(key, tensor)

        listener._reply_pool.shutdown(wait=True)
        assert len(served) == 1
        assert served[0][0] == parked_req
        assert torch.equal(served[0][1], tensor)
        assert key not in listener._parked

    def test_dispatch_parked_without_waiters_is_noop(self):
        listener, cond = self._make_listener()
        served = []
        listener._serve_reply = lambda req, value: served.append((req, value))

        listener.dispatch_parked(("model.norm.output.i0", "reqB"), torch.zeros(2))

        listener._reply_pool.shutdown(wait=True)
        assert served == []

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
        listener = PPListener(
            buffer={}, condition=threading.Condition(), pull_group=MagicMock(),
            local_rank=0, device=torch.device("cpu"),
        )

        import nnsight.modeling.vllm.pp_listener as L

        pending = {}                     # response_tag -> value the producer will return
        request_tags = []                # every tag seen on a request send
        lock = threading.Lock()

        def fake_send(tensor, group=None, group_dst=None, tag=None):
            if tag == L.TAG_REQUEST:
                _rank, rtag, key = L._decode_request(tensor)
                idx = int(key.split("|", 1)[0][3:])     # "req<idx>" -> idx
                with lock:
                    pending[rtag] = float(idx)
                    request_tags.append(rtag)

        def fake_recv(tensor, group=None, group_src=None, tag=None):
            # Two recvs per pull on its private tag: the shape header (int64,
            # _META_SLOTS) then the flat data sized from it.
            with lock:
                val = pending.get(tag)
            assert val is not None, f"recv on tag {tag!r} with no matching request"
            if tensor.dtype == torch.int64 and tensor.numel() == L._META_SLOTS:
                tensor.zero_()
                tensor[0] = 1                                       # one tensor
                tensor[1] = L._DTYPE_TO_CODE[torch.float32]         # true dtype
                tensor[2] = 1                                       # ndim
                tensor[3] = 1                                       # dim 0
            else:
                tensor.fill_(val)

        fake_dist = MagicMock()
        fake_dist.send.side_effect = fake_send
        fake_dist.recv.side_effect = fake_recv

        got = {}

        def pull(i):
            out = listener.pull_from_remote(
                source_rank=1, provider_string=prov, req_id=f"req{i}",
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


class TestMergeCollectedSaves:
    """The per-rank ``collect_nnsight`` result merge (one util, was triplicated;
    the serve copy used a flat ``dict.update`` that clobbered PP sentinels)."""

    def _pack(self, obj):
        import pickle as _pk

        import zstandard

        return zstandard.ZstdCompressor(level=1).compress(_pk.dumps(obj))

    def test_merges_sentinel_slots_across_ranks(self):
        from nnsight.modeling.vllm.collect import merge_collected_saves

        # Each rank owns one slot of a 2-element saved list; the other is a
        # sentinel. A flat dict.update would drop one rank's real value.
        a = {"req0": {"x": [torch.tensor(1.0), NOT_ON_THIS_RANK]}}
        b = {"req0": {"x": [NOT_ON_THIS_RANK, torch.tensor(2.0)]}}
        out = merge_collected_saves([None, self._pack(a), None, self._pack(b)])
        assert set(out) == {"req0"}
        merged = out["req0"]["x"]
        assert merged[0].item() == 1.0 and merged[1].item() == 2.0

    def test_disjoint_base_ids_and_names_combine(self):
        from nnsight.modeling.vllm.collect import merge_collected_saves

        a = {"r0": {"x": torch.tensor(1.0)}}
        b = {"r1": {"y": torch.tensor(2.0)}}
        out = merge_collected_saves([self._pack(a), self._pack(b)])
        assert out["r0"]["x"].item() == 1.0
        assert out["r1"]["y"].item() == 2.0

    def test_all_none_returns_empty(self):
        from nnsight.modeling.vllm.collect import merge_collected_saves

        assert merge_collected_saves([None, None]) == {}


class TestPPTimeoutConstants:
    """PP timeouts live in one place (``pp.py``); only the readiness-gate
    deadline is env-overridable (the one with a real false-trip risk — a slow
    upstream cross-stage pull keeps a worker from reaching its local part)."""

    def test_env_float_parses_override(self, monkeypatch):
        from nnsight.modeling.vllm.pp import _env_float
        monkeypatch.setenv("NNSIGHT_TEST_TO", "12.5")
        assert _env_float("NNSIGHT_TEST_TO", 30.0) == 12.5

    def test_env_float_default_when_absent(self, monkeypatch):
        from nnsight.modeling.vllm.pp import _env_float
        monkeypatch.delenv("NNSIGHT_TEST_TO", raising=False)
        assert _env_float("NNSIGHT_TEST_TO", 30.0) == 30.0

    def test_env_float_falls_back_on_malformed(self, monkeypatch):
        from nnsight.modeling.vllm.pp import _env_float
        monkeypatch.setenv("NNSIGHT_TEST_TO", "not-a-number")
        assert _env_float("NNSIGHT_TEST_TO", 30.0) == 30.0  # default, not a crash

    def test_named_constants_have_documented_defaults(self):
        from nnsight.modeling.vllm import pp
        assert pp.PP_GATE_TIMEOUT_S == 30.0
        assert pp.PP_FINALIZE_JOIN_S == 5.0
        assert pp.PP_GATE_POLL_S == 1e-4
        assert pp.PP_LISTENER_BACKOFF_S == 0.5

    def test_gate_timeout_honors_env_at_import(self):
        # End-to-end wiring: the module-level constant must read the documented
        # env var name (a fresh process, since the constant is import-time).
        import subprocess
        import sys

        def _read(env):
            r = subprocess.run(
                [sys.executable, "-c",
                 "import nnsight.modeling.vllm.pp as p; print(p.PP_GATE_TIMEOUT_S)"],
                capture_output=True, text=True, env=env, timeout=120,
            )
            assert r.returncode == 0, r.stderr[-500:]
            return float(r.stdout.strip().splitlines()[-1])

        base = dict(os.environ)
        base.pop("NNSIGHT_PP_GATE_TIMEOUT", None)
        assert _read(base) == 30.0
        assert _read({**base, "NNSIGHT_PP_GATE_TIMEOUT": "7.5"}) == 7.5


class TestDerivedOwnership:
    """``PPModuleMap`` resolves a module's owning stage from the load-time
    meta exchange (which module is REAL on which rank) — its ONLY mechanism.
    Hardcoded embedding/norm/head name tables miss real architectures
    (Falcon ``word_embeddings``, OPT ``final_layer_norm``, GPT-NeoX
    ``embed_in``/``embed_out``); derived ownership is name-agnostic. The
    runner injects the three structural last-rank entries
    (``logits``/``samples``/``logits_processor``) right after the exchange.
    """

    def _map(self, owners):
        from nnsight.modeling.vllm.pp import PPModuleMap

        m = PPModuleMap(pp_world_size=2)
        m.set_derived_owners(owners)
        return m

    def test_nonstandard_embed_norm_head_resolve(self):
        # Simulated exchange: vLLM ``named_modules`` keys, NON-standard names.
        m = self._map({
            "model.word_embeddings": 0,     # Falcon embedding
            "model.layers.0": 0, "model.layers.1": 0,
            "model.layers.2": 1, "model.layers.3": 1,
            "model.final_layer_norm": 1,    # OPT final norm
            "embed_out": 1,                 # GPT-NeoX head
        })
        assert m.get_owning_rank("model.model.word_embeddings.output") == 0
        assert m.get_owning_rank("model.model.final_layer_norm.output") == 1
        assert m.get_owning_rank("model.embed_out.output") == 1

    def test_layer_submodules_inherit_stage_from_derived_map(self):
        m = self._map({
            "model.layers.0": 0, "model.layers.1": 0,
            "model.layers.2": 1, "model.layers.3": 1,
        })
        # A submodule resolves to its nearest owned ancestor's stage.
        assert m.get_owning_rank("model.model.layers.3.mlp.output") == 1
        assert m.get_owning_rank("model.model.layers.0.self_attn.output") == 0

    def test_structural_last_rank_claims_resolve(self):
        # ``logits_processor`` is built on EVERY rank (real everywhere), so the
        # exchange can't attribute it; the runner injects the structural
        # last-rank claims for it and for nnsight's own root wrapper modules.
        # With the entries installed, all three resolve — including via the
        # provider-string form with an iteration suffix.
        m = self._map({
            "model.layers.0": 0, "model.layers.3": 1,
            "logits": 1, "samples": 1, "logits_processor": 1,
        })
        assert m.get_owning_rank("model.logits_processor.output") == 1
        assert m.get_owning_rank("model.logits.i0") == 1
        assert m.get_owning_rank("model.samples.i0") == 1

    def test_ambiguous_module_without_claim_is_unresolved(self):
        # A module the exchange dropped as ambiguous and nobody claimed
        # resolves to None — treated as local (safe), and a genuine
        # cross-stage consume raises descriptively at pull time.
        m = self._map({"model.layers.0": 0, "model.layers.3": 1})
        assert m.get_owning_rank("model.logits_processor.output") is None

    def test_no_derived_owners_resolves_nothing(self):
        from nnsight.modeling.vllm.pp import PPModuleMap

        # Before the exchange installs owners (nothing traces that early),
        # every path is unresolved -> treated as local.
        m = PPModuleMap(pp_world_size=2)
        assert m.get_owning_rank("model.layers.0.output") is None
        assert m.get_owning_rank("model.embed_tokens.output") is None
        assert m.is_local("model.layers.0.output", local_rank=0) is True


class TestPullErrorReply:
    """A producer that can't serialize a requested value must TELL the blocked
    consumer (an error reply on the per-pull tag), not silently drop it.

    A per-op gloo recv timeout is not usable as a backstop — it closes the
    whole peer pair (probed: ``Application timeout caused pair closure``), so
    the error must come from the producer. Triggers in practice: a dict-valued
    ``.inputs`` cross-stage read, a mixed-dtype tuple (``torch.cat`` fails), a
    value with more tensors/dims than the fixed shape header holds.
    """

    def _listener(self, local_rank=0):
        return PPListener(
            buffer={}, condition=threading.Condition(), pull_group=MagicMock(),
            local_rank=local_rank, device=torch.device("cpu"),
        )

    def test_serve_reply_sends_error_header_for_unserializable_value(self):
        import nnsight.modeling.vllm.pp_listener as L

        listener = self._listener()
        sent = []
        fake_dist = MagicMock()
        fake_dist.send.side_effect = lambda t, **kw: sent.append(t.clone())
        fake_dist.is_initialized.return_value = True

        # A 1-tuple holding a non-tensor (like a kwargs dict): ``.detach()`` fails.
        bad = (torch.zeros(2), {"attention_mask": 1})
        with patch.object(L, "dist", fake_dist):
            listener._serve_reply((1, L.TAG_RESPONSE_BASE), bad)

        assert len(sent) == 2, "expected an error header + message"
        header, msg_buf = sent
        assert int(header[0].item()) == -1, "slot 0 must flag an error"
        assert int(header[1].item()) == msg_buf.numel(), "slot 1 = message length"
        msg = bytes(msg_buf.numpy()).decode("utf-8")
        assert "detach" in msg or "AttributeError" in msg, msg

    def test_serve_reply_error_on_header_overflow(self):
        import nnsight.modeling.vllm.pp_listener as L

        listener = self._listener()
        sent = []
        fake_dist = MagicMock()
        fake_dist.send.side_effect = lambda t, **kw: sent.append(t.clone())
        fake_dist.is_initialized.return_value = True

        # A tuple of many high-rank tensors overflows the 32-slot shape header.
        big = tuple(torch.zeros(1, 1, 1, 1, 1, 1) for _ in range(8))
        with patch.object(L, "dist", fake_dist):
            listener._serve_reply((1, L.TAG_RESPONSE_BASE), big)

        assert int(sent[0][0].item()) == -1
        msg = bytes(sent[1].numpy()).decode("utf-8")
        assert "header" in msg.lower() or "slot" in msg.lower(), msg

    def test_recv_reply_raises_on_error_header(self):
        import nnsight.modeling.vllm.pp_listener as L

        listener = self._listener(local_rank=1)
        err = b"AttributeError: 'tuple' object has no attribute 'detach'"
        header = torch.zeros(L._META_SLOTS, dtype=torch.int64)
        header[0] = -1
        header[1] = len(err)
        seq = [header, torch.frombuffer(bytearray(err), dtype=torch.uint8).clone()]
        state = {"i": 0}

        def fake_recv(tensor, **kw):
            tensor.copy_(seq[state["i"]])
            state["i"] += 1

        fake_dist = MagicMock()
        fake_dist.recv.side_effect = fake_recv
        with patch.object(L, "dist", fake_dist):
            with pytest.raises(RuntimeError, match="owning rank|producer|detach"):
                listener._recv_reply(
                    MagicMock(), 0, L.TAG_RESPONSE_BASE,
                    module_path="model.layers.50",
                )

    def test_clear_buffer_error_replies_parked_pulls(self):
        """A pull still parked when its request finalizes (its value will never
        be produced — e.g. a run-ahead worker pulling past generation end) must
        be error-replied, not silently dropped, so the blocked consumer raises
        and its worker exits instead of leaking."""
        import nnsight.modeling.vllm.pp_listener as L

        listener = self._listener()
        served = []
        listener._serve_error_reply = lambda req, msg: served.append((req, msg))
        # Park a pull for req_id "r7", keyed by the composite (provider, req_id).
        parked_req = (1, L.TAG_RESPONSE_BASE)
        listener._parked[("model.logits.i9", "r7")] = [parked_req]

        listener.clear_buffer(req_ids={"r7"})

        # Pool runs the error reply; drain it.
        listener._reply_pool.shutdown(wait=True)
        assert len(served) == 1, served
        assert served[0][0] == parked_req
        assert "never produced" in served[0][1]
        assert ("model.logits.i9", "r7") not in listener._parked

    def test_serve_reply_roundtrips_normal_value(self):
        """The error path must not regress the normal reply: a real tensor still
        sends header (slot0 = ntensors, NOT -1) then flat data."""
        import nnsight.modeling.vllm.pp_listener as L

        listener = self._listener()
        sent = []
        fake_dist = MagicMock()
        fake_dist.send.side_effect = lambda t, **kw: sent.append(t.clone())
        fake_dist.is_initialized.return_value = True
        with patch.object(L, "dist", fake_dist):
            listener._serve_reply((1, L.TAG_RESPONSE_BASE), torch.arange(6.0))
        assert int(sent[0][0].item()) == 1      # one tensor, not an error
        assert sent[1].numel() == 6

    def test_wire_dtype_sizes_consumer_recv(self):
        """Regression for the sampled-token-ids crash: an integer-valued output
        (int32 token ids from a bf16-weighted model) must round-trip with its
        TRUE dtype. The producer stamps the dtype in the header; the consumer
        sizes its recv buffer from it — a weight-derived guess (bf16, 2 B/elem
        vs int32's 4) under-sized the buffer and gloo aborted."""
        import nnsight.modeling.vllm.pp_listener as L

        producer = self._listener(local_rank=0)
        sent = []
        fake_dist = MagicMock()
        fake_dist.send.side_effect = lambda t, **kw: sent.append(t.clone())
        fake_dist.is_initialized.return_value = True
        token_ids = torch.tensor([[11], [42], [7], [9000]], dtype=torch.int32)
        with patch.object(L, "dist", fake_dist):
            producer._serve_reply((1, L.TAG_RESPONSE_BASE), token_ids)

        header, flat = sent
        assert int(header[1].item()) == L._DTYPE_TO_CODE[torch.int32]

        # Feed the producer's exact wire bytes to a consumer.
        consumer = self._listener(local_rank=1)
        seq = [header, flat]
        state = {"i": 0}

        def fake_recv(tensor, **kw):
            tensor.copy_(seq[state["i"]])
            state["i"] += 1

        fake_dist2 = MagicMock()
        fake_dist2.recv.side_effect = fake_recv
        with patch.object(L, "dist", fake_dist2):
            out = consumer._recv_reply(
                MagicMock(), 0, L.TAG_RESPONSE_BASE,
                module_path="sampler.output_tokens",
            )
        assert out.dtype == torch.int32
        assert torch.equal(out.cpu(), token_ids)

    def test_unencodable_dtype_error_replies(self):
        """A dtype outside the wire codec cannot be sized by the consumer; the
        producer must error-reply, never stamp an unknown code."""
        import nnsight.modeling.vllm.pp_listener as L

        listener = self._listener()
        sent = []
        fake_dist = MagicMock()
        fake_dist.send.side_effect = lambda t, **kw: sent.append(t.clone())
        fake_dist.is_initialized.return_value = True
        odd = torch.zeros(3, dtype=torch.complex32)  # not in the codec
        with patch.object(L, "dist", fake_dist):
            listener._serve_reply((1, L.TAG_RESPONSE_BASE), odd)

        assert int(sent[0][0].item()) == -1, "must be an error header"
        msg = bytes(sent[1].numpy()).decode("utf-8")
        assert "dtype" in msg.lower(), msg


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

    def test_is_pp_missing_cached_per_lookup(self):
        """The remote status is constant for the model's life, so
        ``_is_pp_missing`` must compute it once per lookup and cache it (the
        ``get_owning_rank``/``is_local`` walk runs on every per-token access)."""
        import torch.nn as nn

        from nnsight.intervention.interleaver import Interleaver
        from nnsight.modeling.vllm.pp_envoy import PPEnvoy, _is_pp_missing

        interleaver = Interleaver()
        interleaver.pp_enabled = True
        interleaver.pp_local_rank = 0
        interleaver.pp_listener = None
        mock_map = MagicMock()
        mock_map.is_local.return_value = False  # remote
        interleaver.pp_module_map = mock_map
        interleaver.pp_module_meta = {}

        # NON-stub module so resolution goes through the costly is_local walk
        # (a PPMissingLayer would short-circuit before it).
        envoy = PPEnvoy(nn.Identity(), path="model.logits", interleaver=interleaver)

        assert _is_pp_missing(envoy, "output") is True
        assert _is_pp_missing(envoy, "output") is True
        assert _is_pp_missing(envoy, "output") is True
        assert mock_map.is_local.call_count == 1  # computed once, then cached
        assert interleaver.__dict__["_pp_remote_cache"]["model.logits.output"] is True

        # A different key is a distinct cache entry (computed once more).
        assert _is_pp_missing(envoy, "input") is True
        assert mock_map.is_local.call_count == 2

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
        from nnsight.intervention.interleaver import (
            PPWorkerProgress,
            _active_mediator,
        )
        from nnsight.intervention.tracing.iterator import IteratorTracer

        my_mediator = MagicMock(name="my_mediator")
        other_mediator = MagicMock(name="other_mediator")
        my_mediator.pp_progress = PPWorkerProgress()
        other_mediator.pp_progress = PPWorkerProgress()
        my_mediator.pp_progress.worker_iteration = "untouched"
        other_mediator.pp_progress.worker_iteration = "untouched"

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
            assert my_mediator.pp_progress.worker_iteration == 0, (
                "iterator bound to interleaver.current instead of the "
                "worker thread-local mediator"
            )
            # ...and the other invoke's mediator must be untouched.
            assert other_mediator.pp_progress.worker_iteration == "untouched"
            gen.close()
        finally:
            _active_mediator.value = None

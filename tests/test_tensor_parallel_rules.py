"""The tensor-parallel rule table, checked without a GPU.

`tests/tp/test_sharded_tracing.py` proves the gather is *correct*, but it needs
two GPUs and so does not run everywhere. These tests cover the part that
goes wrong quietly and can be checked anywhere: whether the table still describes
the transformers it is running against, and whether the cases it refuses actually
raise.

The failure they exist for is version drift. transformers owns the list of
parallel styles; nnsight's table has to keep up with it, and a style that appears
upstream without a rule here is only discovered when someone deploys a model that
uses it.
"""

from __future__ import annotations


import pytest
import torch

from nnsight.intervention.interleaver import Interleaver
from nnsight.modeling.tp import (
    SIDES,
    UNSUPPORTED,
    TPFragments,
    UnsupportedParallelStyle,
)


def _upstream_styles() -> set:
    # The module `TPFragments._style_of` imports, so the table is checked against
    # the registry the supported backend actually reads. On transformers below the
    # 5.16 floor this import is what fails, which is the point.
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

    return set(ALL_PARALLEL_STYLES._global_mapping)


class _FakeMesh:
    def __init__(self, size: int = 2) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeInterleaver:
    """Weakref-able stand-in.

    `install_controller` registers a route on the envoy's interleaver, and a
    weakref needs a real object to point at. Nothing here is called: these tests
    instrument a tree, they do not run one.
    """


class _FakeEnvoy:
    """What `instrument` reads off an envoy."""

    def __init__(self, module: torch.nn.Module, path: str = "model.thing") -> None:
        self._module = module
        self.path = path
        self.interleaver = _FakeInterleaver()


ROOT = "model"


def _tp_wrapped_forward():
    """A stand-in for transformers' TP forward wrapper, recognised by qualname."""

    class TensorParallelLayer:
        @staticmethod
        def install_forward():
            def tp_forward(*args, **kwargs):
                raise AssertionError("these tests instrument a tree, they do not run one")

            return tp_forward

    return TensorParallelLayer.install_forward()


def _sharded(style: str | None, path: str = "model.thing", size: int = 2, **attrs):
    """A `TPFragments` that has walked a model sharding ``path`` as ``style``.

    Built the way transformers describes a sharded model: the plan lives on the
    *root*, keyed by each module's path below it, with a mesh saying it was
    really split — so the root is instrumented first, exactly as `Envoy.__init__`
    does before walking its children.
    """
    root = torch.nn.Identity()
    root.tp_plan = {path[len(ROOT) + 1 :]: style} if style is not None else {}
    root._device_mesh = _FakeMesh(size)

    # Stands in for the wrapper transformers installs when it shards a module:
    # `instrument` records a rule only for a module it actually wrapped.
    module = torch.nn.Identity()
    module.__dict__["forward"] = _tp_wrapped_forward()
    for name, value in attrs.items():
        setattr(module, name, value)

    fragments = TPFragments()
    fragments.instrument(_FakeEnvoy(root, path=ROOT))
    fragments.instrument(_FakeEnvoy(module, path=path))
    return fragments


class TestStyleCoverage:
    """Every style transformers knows about has a rule here, and vice versa."""

    def test_no_upstream_style_is_unaccounted_for(self):
        # The one that matters: a new style upstream with no rule here is a model
        # nnsight will refuse at deploy time. Better to find out at test time.
        missing = _upstream_styles() - (set(SIDES) | set(UNSUPPORTED))
        assert not missing, (
            f"transformers has parallel styles this version has no rule for: "
            f"{sorted(missing)}. Add each to SIDES (with the sides that "
            f"carry a shard) or to UNSUPPORTED."
        )

    def test_no_rule_names_a_style_that_no_longer_exists(self):
        stale = (set(SIDES) | set(UNSUPPORTED)) - _upstream_styles()
        assert not stale, (
            f"rules name parallel styles transformers no longer has: "
            f"{sorted(stale)} — probably renamed upstream."
        )


class TestInstrument:
    """What the interleaver does as the Envoy tree is built."""

    def test_starts_inert(self):
        interleaver = TPFragments()
        assert not interleaver.enabled
        assert not interleaver.tp_rules

    def test_an_unsharded_module_records_nothing(self, monkeypatch):
        interleaver = _sharded(None)

        assert not interleaver.enabled
        assert not interleaver.tp_rules

    def test_a_sharded_module_records_its_side_and_enables(self, monkeypatch):
        interleaver = _sharded("colwise", path="model.q_proj")

        assert interleaver.enabled
        assert "model.q_proj.output" in interleaver.tp_rules

    def test_a_row_parallel_module_records_its_input(self, monkeypatch):
        interleaver = _sharded("rowwise", path="model.o_proj")

        # At the handoff a row-parallel input is already this rank's slice and its
        # output is this rank's partial sum — its own post-transform reduces it
        # after us. Recorded as the DTensor placements that say so, which is what
        # the gather asserts onto a value that arrives without one.
        from torch.distributed.tensor import Partial, Shard

        assert interleaver.tp_rules["model.o_proj.input"][1] == Shard(-1)
        assert isinstance(interleaver.tp_rules["model.o_proj.output"][1], Partial)

    @pytest.mark.parametrize("style", sorted(UNSUPPORTED))
    def test_a_refused_style_raises(self, style, monkeypatch):
        with pytest.raises(UnsupportedParallelStyle, match=style):
            _sharded(style)

    def test_an_unknown_style_raises(self, monkeypatch):
        with pytest.raises(UnsupportedParallelStyle, match="not a parallel style"):
            _sharded("something_new_upstream")


def test_an_unsharded_model_is_never_asked():
    # The interleaver checks `enabled` before anything else, so an unsharded
    # model never reaches `read` at all.
    from nnsight.modeling.tp import TPFragments

    assert TPFragments().enabled is False




class TestMaxTpSize:
    """The largest degree a checkpoint's config says it splits into.

    Divisibility is the whole constraint: transformers shards attention by head
    and the MLP by its intermediate dimension, and its all-gather assumes equal
    pieces — an uneven degree does not run slower, it does not run.
    """

    def _config(self, **fields):
        plan = fields.pop("plan", {"layers.*.self_attn.q_proj": "colwise"})
        config = type("Config", (), {})()
        config.base_model_tp_plan = plan
        for name, value in fields.items():
            setattr(config, name, value)
        return config

    def test_the_gcd_of_the_dimensions_it_must_divide(self):
        from nnsight.modeling.tp import max_tp_size

        # 24 heads, 8 kv heads, 8192 intermediate -> 8.
        assert max_tp_size(self._config(
            num_attention_heads=24, num_key_value_heads=8, intermediate_size=8192
        )) == 8

    def test_a_low_key_value_head_count_caps_it(self):
        from nnsight.modeling.tp import max_tp_size

        # Qwen2.5-0.5B's shape: plenty of heads, but 2 kv heads stops it at 2.
        assert max_tp_size(self._config(
            num_attention_heads=14, num_key_value_heads=2, intermediate_size=4864
        )) == 2

    def test_no_plan_means_it_cannot_be_split(self):
        from nnsight.modeling.tp import max_tp_size

        assert max_tp_size(self._config(plan=None, num_attention_heads=12)) is None

    def test_an_unsupported_style_in_the_plan_refuses(self):
        from nnsight.modeling.tp import max_tp_size

        # A model whose plan nnsight cannot gather would fail at load; it must
        # not be *placed* as though it could be split.
        assert max_tp_size(self._config(
            plan={"layers.*.mlp.experts": "megamoe_experts"},
            num_attention_heads=16, num_key_value_heads=16, intermediate_size=4096,
        )) is None

    def test_an_odd_dimension_leaves_nothing_to_split(self):
        from nnsight.modeling.tp import max_tp_size

        assert max_tp_size(self._config(
            num_attention_heads=3, num_key_value_heads=1, intermediate_size=11
        )) is None

    def test_dimensions_are_read_from_a_nested_text_config(self):
        from nnsight.modeling.tp import max_tp_size

        # A multimodal config keeps the transformer dims one level down; reading
        # the outer one finds nothing to divide and would call every degree fine.
        outer = self._config()
        outer.text_config = self._config(
            num_attention_heads=32, num_key_value_heads=8, intermediate_size=14336
        )
        assert max_tp_size(outer) == 8


class TestBytesPerElement:
    """Sizing a checkpoint by the dtype it will be held in."""

    def test_torch_dtypes(self):
        from nnsight.modeling.mixins.remotable import bytes_per_element

        assert bytes_per_element("bfloat16") == 2
        assert bytes_per_element("torch.float32") == 4
        assert bytes_per_element("float64") == 8

    def test_sub_byte_quantizations_beat_torchs_rounding(self):
        from nnsight.modeling.mixins.remotable import bytes_per_element

        # torch.int4 exists and reports itemsize 1 -- the smallest it can
        # address -- so trusting it would size 4-bit weights at twice reality.
        assert bytes_per_element("int4") == 0.5
        assert bytes_per_element("nf4") == 0.5
        assert bytes_per_element("int8") == 1

    def test_an_unknown_name_raises_rather_than_guessing(self):
        from nnsight.modeling.mixins.remotable import bytes_per_element

        with pytest.raises(ValueError, match="Unknown dtype"):
            bytes_per_element("float3")


class TestTransformersVersionFloor:
    """Sharding is refused on a transformers whose TP nnsight cannot read.

    The floor is the DTensor rewrite. Below it the plan is stamped per module
    rather than kept on the model, so nnsight finds nothing sharded — which does
    not fail, it hands intervention code one rank's slice as the whole tensor.
    That is the same shape of bug as the one caught on a live deployment running
    5.14.1, where a tied-embedding model returned logits four times the
    vocabulary width while the argmax — and so every eyeball check — stayed right.
    """

    def _check(self, monkeypatch, version: str):
        import transformers

        from nnsight.modeling.tp import fragments as tp_fragments

        monkeypatch.setattr(transformers, "__version__", version)
        monkeypatch.setattr(tp_fragments, "_version_checked", False)
        tp_fragments._check_transformers_version()

    def test_an_older_transformers_is_refused(self, monkeypatch):
        from nnsight.modeling.tp import UnsupportedTransformersVersion

        with pytest.raises(UnsupportedTransformersVersion, match="5.16"):
            self._check(monkeypatch, "5.15.1")

    @pytest.mark.parametrize(
        "version", ["5.16.0", "5.16.1", "6.0.0", "5.16.0.dev0", "5.17.0rc1"]
    )
    def test_the_floor_and_above_are_allowed(self, monkeypatch, version):
        # Including pre-releases of the fixed series: an editable transformers
        # checkout reports 5.16.0.dev0, which a plain >= would reject.
        self._check(monkeypatch, version)

    def test_it_only_runs_once(self, monkeypatch):
        # instrument() calls this per sharded module -- hundreds on a real model
        # -- so the import and version parse have to happen once, not per call.
        import transformers

        from nnsight.modeling.tp import fragments as tp_fragments

        monkeypatch.setattr(transformers, "__version__", "5.16.0")
        monkeypatch.setattr(tp_fragments, "_version_checked", False)
        tp_fragments._check_transformers_version()
        monkeypatch.setattr(transformers, "__version__", "0.1.0")
        tp_fragments._check_transformers_version()  # would raise if re-checked


class TestTheTwoGatesAgree:
    """Placement and load must refuse the same set of models.

    `max_tp_size` decides whether a model is *placed* tensor-parallel;
    `instrument` decides whether it can be *traced* that way. They ran different
    predicates -- one checked UNSUPPORTED, the other checked SIDES -- so a
    style in neither list passed placement and raised at load, after a server had
    already allocated the cards and read the weights onto them.
    """

    def _config(self, plan):
        class Config:
            base_model_tp_plan = plan
            num_attention_heads = 32
            num_key_value_heads = 8
            intermediate_size = 14336

        return Config()

    def test_a_style_in_neither_list_is_refused(self):
        from nnsight.modeling.tp import SIDES, UNSUPPORTED, max_tp_size

        # A plan naming something no list covers. `colwise_rep` used to stand here
        # — it was Llama-4's plan and no version registered it — but 5.16 added it
        # to ALL_PARALLEL_STYLES and it now has a rule, so the case needs a name
        # that is still genuinely unknown.
        unknown = "colwise_upside_down"
        assert unknown not in SIDES
        assert unknown not in UNSUPPORTED
        assert max_tp_size(self._config({"layer.q_proj": unknown})) is None

    def test_a_refused_style_is_still_refused(self):
        from nnsight.modeling.tp import max_tp_size

        assert max_tp_size(self._config({"layer.experts": "megamoe_experts"})) is None

    def test_a_known_style_still_places(self):
        from nnsight.modeling.tp import max_tp_size

        assert max_tp_size(self._config({"layer.q_proj": "colwise"})) == 8

    def test_every_placeable_plan_is_instrumentable(self):
        # The property, stated directly: anything max_tp_size accepts, instrument
        # must not raise on. Both now read SIDES, so this holds by
        # construction -- it is here to fail if they ever diverge again.
        from nnsight.modeling.tp import SIDES, max_tp_size

        for style in SIDES:
            assert max_tp_size(self._config({"layer.x": style})) == 8, style


class TestExpertParallelIsNotAutomaticallyRefused:
    """`moe_tp_experts` needs no gather, and refusing it cost the MoE models.

    It is expert-parallel, which is why it was refused on sight. But its forward
    all-reduces -- `_prepare_input_fn` applies only `all_reduce_backward`, which
    is identity going forwards -- so both sides arrive whole, exactly like
    `all_reduce`. 26 of the configs shipped with transformers 5.15 (Mixtral,
    DeepSeek-V3, Qwen3-MoE, ...) were refused for this and nothing else.
    """

    def test_it_is_no_longer_refused(self):
        from nnsight.modeling.tp import UNSUPPORTED

        assert "moe_tp_experts" not in UNSUPPORTED

    def test_its_output_is_reduced_not_gathered(self):
        from nnsight.modeling.tp import SIDES

        assert SIDES["moe_tp_experts"] == {"output": "partial"}

    def test_an_moe_plan_can_now_be_placed(self):
        from nnsight.modeling.tp import max_tp_size

        class Config:
            base_model_tp_plan = {
                "layers.*.self_attn.q_proj": "colwise",
                "layers.*.mlp.experts": "moe_tp_experts",
            }
            num_attention_heads = 32
            num_key_value_heads = 8
            intermediate_size = 14336

        assert max_tp_size(Config()) == 8

    def test_a_style_no_model_has_exercised_is_still_refused(self):
        """The refused set is now what nothing has been run against.

        ``ep_router`` and ``grouped_gemm`` left it because an expert-parallel
        model was actually traced end to end (`tests/tp/test_cpu_expert_parallel.py`)
        and their values proved to need no gather at all. The rest stay refused
        for the reason the table has always given: reading a style's transforms
        is not the same as having run one.
        """
        from nnsight.modeling.tp import SIDES, UNSUPPORTED

        assert "megamoe_experts" in UNSUPPORTED
        assert "mla_kv_a_proj" in UNSUPPORTED
        assert "ep_router" in SIDES
        assert "grouped_gemm" in SIDES


class FakeMesh:
    """Just enough device mesh for the pure-tensor helpers."""

    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class TestSplitStandsAlone:
    """`split` cuts a value down with no gather to reverse.

    `TPEnvoy.__call__` uses it to prepare the argument of an ad-hoc call on a
    sharded module. That value is whole because the caller is holding it, not
    because anything assembled it, so there is no gather to undo and the
    location's rule is the only thing to go on.

    It used to be served by `fragment`, the same method that reversed a gather —
    which meant an ad-hoc call made while its own location's handoff was still
    open consumed the record that handoff was going to use, and the module then
    ran sharded weights against a whole tensor.
    """

    def _split_calls(self, monkeypatch):
        from nnsight.modeling.tp import fragments as tp_fragments

        seen = []
        monkeypatch.setattr(
            tp_fragments,
            "_fragment_tensor",
            lambda tensor, *args, **kwargs: (seen.append(tensor), tensor)[1],
        )
        return seen

    def test_it_uses_the_rule(self, monkeypatch):
        fragments = _sharded("rowwise", path="model.o_proj")
        split = self._split_calls(monkeypatch)

        fragments.split("model.o_proj.input", torch.randn(1, 4, 8))

        assert len(split) == 1, "a standalone split left the value whole"

    def test_a_location_with_no_rule_is_left_alone(self, monkeypatch):
        fragments = _sharded("rowwise", path="model.o_proj")
        split = self._split_calls(monkeypatch)

        fragments.split("model.o_proj.something_else", torch.randn(1, 4, 8))

        assert not split

    def test_whole_hands_back_its_own_way_out(self, monkeypatch):
        """The reversal is a closure, not a record another call could consume."""
        fragments = _sharded("rowwise", path="model.o_proj")
        monkeypatch.setattr(
            "nnsight.modeling.tp.fragments._gather", lambda value, *a: value
        )
        _, undo = fragments.whole("model.o_proj.input", torch.randn(1, 4, 8))

        assert callable(undo)
        # Nothing was written anywhere for a later call to find and take.
        assert not [name for name in vars(fragments) if "pending" in name]


class TestOnlyBoundariesAreReassembled:
    """A module's two sides are made whole; everything inside a forward is not.

    An operation's value has no rule and nothing on it says which axis holds its
    shard once it has left the module that produced it — and the axis moves, so
    there is nothing safe to assume. It is handed over as this device's piece and
    the trace reassembles it with `gather`/`shard`, naming the axis. `Envoy.source`
    warns on a sharded model so that is not a surprise.
    """

    def test_a_module_side_with_a_rule_is_reassembled(self):
        fragments = _sharded("colwise", path="model.q_proj")

        # colwise records only its output; its input is whole where we stand.
        assert fragments.fragmented("model.q_proj.output")
        assert not fragments.fragmented("model.q_proj.input")

    def test_an_operation_inside_a_forward_is_not(self):
        fragments = _sharded("colwise", path="model.q_proj")

        assert not fragments.fragmented("model.q_proj.source.F_linear_0.output")
        assert not fragments.fragmented("model.q_proj.source.F_linear_0.input")

    def test_a_module_with_no_rule_is_not(self):
        """`act_fn` between a colwise output and a rowwise input, say."""
        fragments = _sharded("colwise", path="model.q_proj")

        assert not fragments.fragmented("model.act_fn.output")

    def test_a_boundary_without_a_placement_still_uses_its_rule(self):
        """A side is reassembled from its rule even when the value says nothing."""
        from nnsight.modeling.tp import fragments as tp_fragments

        fragments = _sharded("colwise", path="model.q_proj")
        gathered = []
        real = tp_fragments._gather
        try:
            tp_fragments._gather = lambda value, *a: (gathered.append(value), value)[1]
            fragments.whole("model.q_proj.output", torch.randn(1, 4, 8))
        finally:
            tp_fragments._gather = real
        assert len(gathered) == 1


class TestReshardGuard:
    """What goes back to the model must be guarded like what came out of it.

    `_reshard`'s output is consumed by the model's own forward, so splitting a
    value that was never a fragment hands every rank a slice of something whole.
    Divisibility alone cannot tell the two apart: an integer `position_ids` of
    width 8 at tp=4 divides perfectly, and the result is wrong answers on every
    rank identically -- not a hang, and nothing to notice.

    Spies on nnsight's own two primitives rather than on a transformers helper.
    The previous version patched ``transformers.integrations.tensor_parallel.split``,
    which 5.16 deleted -- so these tests broke on a release that changed nothing
    about the behaviour they describe.
    """

    def _calls(self, monkeypatch, name):
        """Record every tensor handed to fragments.``name``, and pass it through."""
        from nnsight.modeling.tp import fragments

        seen = []
        monkeypatch.setattr(
            fragments, name, lambda tensor, *args, **kwargs: (seen.append(tensor), tensor)[1]
        )
        return seen

    @pytest.mark.parametrize(
        "tensor,why",
        [
            (torch.arange(8).reshape(1, 8), "integer position_ids"),
            (torch.zeros(1, 1, 8, 8, dtype=torch.bool), "a boolean mask"),
            (torch.tensor(1.0), "a 0-dim scalar"),
        ],
    )
    def test_a_value_the_gather_skipped_is_not_split(self, monkeypatch, tensor, why):
        from nnsight.modeling.tp.fragments import _gather, _placement, _reshard

        mesh = FakeMesh(4)
        shard = _placement("shard")

        gathered = self._calls(monkeypatch, "_whole_tensor")
        _gather(tensor, mesh, shard)
        assert not gathered, f"{why} should not have been gathered"

        split = self._calls(monkeypatch, "_fragment_tensor")
        _reshard(tensor, mesh, shard, False)
        assert not split, f"{why} was split though it was never gathered"

    def test_a_real_shard_still_round_trips(self, monkeypatch):
        from nnsight.modeling.tp.fragments import _placement, _reshard

        split = self._calls(monkeypatch, "_fragment_tensor")
        _reshard(torch.randn(1, 4, 8), FakeMesh(4), _placement("shard"), False)
        assert len(split) == 1

    def test_a_partial_is_reduced_whatever_its_width(self, monkeypatch):
        """A partial is a whole-width term of a sum, so divisibility is irrelevant.

        Guarding it with the shard's ``% world_size`` test would silently skip the
        re-fragment on any width that does not divide, and the model's own reduce
        would then sum the whole tensor once per rank.
        """
        from nnsight.modeling.tp.fragments import _placement, _reshard

        split = self._calls(monkeypatch, "_fragment_tensor")
        _reshard(torch.randn(1, 4, 7), FakeMesh(4), _placement("partial"), False)
        assert len(split) == 1


class TestHubFailuresAreDistinguishable:
    """"Couldn't read it" must not be reported as "there's nothing to read".

    Every reader on the placement path returns None to mean a real absence -- no
    published parameter count, no config, no sharding plan. None is also what
    `max_tp_size` returns to mean "this model cannot be split at all". So a
    config that failed to download used to become a fact about the architecture,
    and a perfectly shardable model was placed layer-by-layer with a debug line
    as the only trace.
    """

    def _http(self, error_class, status):
        # These carry the response they were built from; a 4xx is the Hub
        # answering and a 5xx is the Hub failing, which is the distinction.
        import requests
        from huggingface_hub.errors import HfHubHTTPError  # noqa: F401

        response = requests.Response()
        response.status_code = status
        return error_class("boom", response=response)

    def test_a_missing_repo_is_an_absence(self):
        from huggingface_hub.errors import RepositoryNotFoundError

        from nnsight.modeling.huggingface import _unreachable

        assert _unreachable(self._http(RepositoryNotFoundError, 404)) is False

    def test_the_hub_failing_is_a_failure_to_read(self):
        from huggingface_hub.errors import HfHubHTTPError

        from nnsight.modeling.huggingface import _unreachable

        assert _unreachable(self._http(HfHubHTTPError, 503)) is True

    def test_the_hub_refusing_is_an_answer(self):
        from huggingface_hub.errors import HfHubHTTPError

        from nnsight.modeling.huggingface import _unreachable

        assert _unreachable(self._http(HfHubHTTPError, 401)) is False

    def test_no_network_and_no_cache_is_a_failure_to_read(self):
        # Reads backwards, which is the whole reason this function exists: the
        # Hub raises LocalEntryNotFoundError when it could not reach the network
        # *and* found nothing cached. It is connectivity, not a missing file.
        from huggingface_hub.errors import LocalEntryNotFoundError

        from nnsight.modeling.huggingface import _unreachable

        assert _unreachable(LocalEntryNotFoundError("offline")) is True

    def test_offline_mode_is_a_failure_to_read(self):
        from huggingface_hub.errors import OfflineModeIsEnabled

        from nnsight.modeling.huggingface import _unreachable

        assert _unreachable(OfflineModeIsEnabled("offline")) is True

    def test_a_socket_error_is_a_failure_to_read(self):
        from nnsight.modeling.huggingface import _unreachable

        assert _unreachable(OSError("connection reset")) is True

    def test_an_unrelated_error_is_not(self):
        from nnsight.modeling.huggingface import _unreachable

        assert _unreachable(ValueError("bad config")) is False


class TestHubCallsAreBounded:
    """A slow Hub gives up rather than parking whoever asked.

    Mostly not this code's doing, which is worth recording because it looked like
    a gap: `AutoConfig.from_pretrained` takes no timeout argument, but everything
    huggingface_hub fetches is bounded by `HF_HUB_ETAG_TIMEOUT` and
    `HF_HUB_DOWNLOAD_TIMEOUT`, both 10s by default. So the config read is bounded
    per request whether or not nnsight says so, and the only call that needed a
    timeout passed explicitly is `model_info`, which accepts one.

    An earlier version of this ran the read on an abandoned daemon thread to
    bound it. That is worse than the problem: a thread left blocked in an
    uninterruptible wait stops the *process* exiting -- verified, the interpreter
    hangs -- which in a model actor means one that will not shut down when told.
    """

    def test_the_transport_is_bounded_by_default(self):
        import huggingface_hub.constants as constants

        assert constants.HF_HUB_ETAG_TIMEOUT > 0
        assert constants.HF_HUB_DOWNLOAD_TIMEOUT > 0

    def test_the_hub_api_call_is_given_a_timeout(self):
        import inspect

        from nnsight.modeling.huggingface import HUB_TIMEOUT_SECONDS, HuggingFaceModel

        source = inspect.getsource(HuggingFaceModel._hub_parameter_count)
        assert "timeout=HUB_TIMEOUT_SECONDS" in source
        assert HUB_TIMEOUT_SECONDS > 0

    def test_nothing_here_abandons_a_thread(self):
        # The regression guard for the above: no background thread on this path.
        import inspect

        import nnsight.modeling.huggingface as hf

        source = inspect.getsource(hf)
        assert "ThreadPoolExecutor" not in source
        assert "daemon=True" not in source


class TestConfigIsReadOnce:
    """Several questions are answered from one config; it is fetched once.

    Sizing a model, working out how it shards, and reporting it in a status all
    read the same object, and each used to fetch it again -- two network round
    trips per cold entry, on the path that was already blocking the event loop.
    """

    def test_a_second_read_does_not_hit_the_network(self, monkeypatch):
        import nnsight.modeling.huggingface as hf

        calls = []

        class Config:
            base_model_tp_plan = {"layers.*.q_proj": "colwise"}
            num_attention_heads = 32
            num_key_value_heads = 8
            intermediate_size = 14336

        def counted(*args, **kwargs):
            calls.append(1)
            return Config()

        import transformers

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", counted)

        key = '{"repo_id": "x", "revision": null}'
        hf._CONFIG_CACHE.pop((key, False), None)
        first = hf.HuggingFaceModel._config(key, False)
        second = hf.HuggingFaceModel._config(key, False)

        assert first is second
        assert len(calls) == 1, f"the config was fetched {len(calls)} times"

    def test_the_two_questions_that_need_it_share_one_read(self, monkeypatch):
        import nnsight.modeling.huggingface as hf

        calls = []

        class Config:
            base_model_tp_plan = {"layers.*.q_proj": "colwise"}
            num_attention_heads = 32
            num_key_value_heads = 8
            intermediate_size = 14336

        def counted(*args, **kwargs):
            calls.append(1)
            return Config()

        key = '{"repo_id": "shared", "revision": null}'
        hf._CONFIG_CACHE.pop((key, False), None)

        import transformers

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", counted)
        # Not what this is about: stub the Hub record so only the config read is
        # being counted.
        monkeypatch.setattr(
            hf.HuggingFaceModel, "_hub_parameter_count", staticmethod(lambda *a: 1_000)
        )

        assert hf.HuggingFaceModel._remoteable_max_tp_size(key) == 8
        described = hf.HuggingFaceModel._remoteable_describe_checkpoint(key, "bfloat16")
        assert described.config is not None
        assert len(calls) == 1, f"the config was fetched {len(calls)} times"


class TestDescribeCheckpoint:
    """One call for what a checkpoint is; a separate one for what a runtime can
    do with it.

    The split is the point. `max_tp_size` is not a property of the files -- the
    same weights shard eight ways under transformers tensor parallelism and not
    at all under something else -- so folding it into a description of the
    checkpoint would make it read as one.
    """

    def test_it_is_not_a_field_of_the_description(self):
        from nnsight.modeling.mixins.remotable import CheckpointInfo

        assert not hasattr(CheckpointInfo(), "max_tp_size")

    def test_every_field_defaults_to_not_knowing(self):
        from nnsight.modeling.mixins.remotable import CheckpointInfo

        info = CheckpointInfo()
        assert (info.size_bytes, info.n_params, info.config, info.revision) == (
            None,
            None,
            None,
            None,
        )

    def test_the_base_wrapper_answers_only_the_size(self, monkeypatch):
        from nnsight.modeling.mixins.remotable import CheckpointInfo, Remotable

        monkeypatch.setattr(
            Remotable, "_remoteable_estimate_bytes", classmethod(lambda cls, *a, **k: 42)
        )
        info = Remotable._remoteable_describe_checkpoint("k", "bfloat16")

        assert isinstance(info, CheckpointInfo)
        assert info.size_bytes == 42
        assert info.config is None and info.revision is None

    def test_a_huggingface_key_reports_its_revision(self, monkeypatch):
        import nnsight.modeling.huggingface as hf

        key = '{"repo_id": "r", "revision": "abc123"}'
        hf._CONFIG_CACHE[(key, False)] = None
        monkeypatch.setattr(
            hf.HuggingFaceModel, "_hub_parameter_count", staticmethod(lambda *a: 1_000)
        )

        info = hf.HuggingFaceModel._remoteable_describe_checkpoint(key, "bfloat16")

        assert info.revision == "abc123"
        assert info.n_params == 1_000
        assert info.size_bytes == 2_000  # bfloat16

    def test_sizing_and_describing_cannot_disagree(self, monkeypatch):
        # One of these decides where a model goes. Two implementations of "how
        # big is it" could drift; there is only one.
        import nnsight.modeling.huggingface as hf

        key = '{"repo_id": "r2", "revision": null}'
        hf._CONFIG_CACHE[(key, False)] = None
        monkeypatch.setattr(
            hf.HuggingFaceModel, "_hub_parameter_count", staticmethod(lambda *a: 7_777)
        )

        described = hf.HuggingFaceModel._remoteable_describe_checkpoint(key, "float32")
        estimated = hf.HuggingFaceModel._remoteable_estimate_bytes(key, "float32")

        assert described.size_bytes == estimated


class TestCheckTpRequest:
    """Refusing a degree the checkpoint cannot actually be split into.

    transformers does not check this and does not fail. Asked to shard a model
    with no plan it shards *nothing*: `verify_tp_plan` returns early on a `None`
    plan and `apply_tensor_parallelism` installs no hooks, so every rank loads a
    complete copy of the weights, nothing warns, and the model answers correctly
    off one rank while the other cards hold redundant copies. The only symptom is
    n times the memory for one model's worth of work, which is why this refuses
    rather than reports.
    """

    def _config(self, **fields):
        plan = fields.pop("plan", {"layers.*.self_attn.q_proj": "colwise"})
        config = type("Config", (), {})()
        config.base_model_tp_plan = plan
        for name, value in fields.items():
            setattr(config, name, value)
        return config

    def _shardable(self):
        # 24 heads, 8 kv heads, 8192 intermediate -> splits 8 ways.
        return self._config(
            num_attention_heads=24, num_key_value_heads=8, intermediate_size=8192
        )

    def test_asking_for_no_degree_checks_nothing(self):
        # The ordinary single-GPU path reaches this on every load; it must not
        # even look at the config.
        from nnsight.modeling.tp import check_tp_request

        check_tp_request(None, None)

    def test_a_checkpoint_with_no_plan_is_refused(self):
        # gpt2's shape, and the case this exists for.
        from nnsight.modeling.tp import UnshardableCheckpoint, check_tp_request

        with pytest.raises(UnshardableCheckpoint, match="cannot be split"):
            check_tp_request(self._config(plan=None, num_attention_heads=12), 2)

    def test_the_refusal_says_what_would_have_happened(self):
        # A message that only says "unsupported" leaves the operator to discover
        # that it *would* have loaded, wrongly. This is the part worth keeping.
        from nnsight.modeling.tp import UnshardableCheckpoint, check_tp_request

        with pytest.raises(UnshardableCheckpoint, match="whole copy of it onto every rank"):
            check_tp_request(self._config(plan=None, num_attention_heads=12), 4)

    def test_a_workable_degree_is_allowed(self):
        from nnsight.modeling.tp import check_tp_request

        for degree in (2, 4, 8):
            check_tp_request(self._shardable(), degree)

    def test_a_degree_that_does_not_divide_is_refused(self):
        # Not a slower option: the all-gather assumes equal pieces, so an uneven
        # degree returns the wrong shape rather than failing outright.
        from nnsight.modeling.tp import UnshardableCheckpoint, check_tp_request

        with pytest.raises(UnshardableCheckpoint, match="splits at most 8 ways"):
            check_tp_request(self._shardable(), 3)

    def test_it_lists_the_degrees_that_would_work(self):
        from nnsight.modeling.tp import UnshardableCheckpoint, check_tp_request

        with pytest.raises(UnshardableCheckpoint, match=r"\[2, 4, 8\]"):
            check_tp_request(self._shardable(), 6)

    def test_it_agrees_with_max_tp_size(self):
        # The two must refuse the same set. `max_tp_size` is what a server places
        # from; this is what a load refuses on. A model placed on cards it then
        # refuses to load onto is the failure this pairing prevents.
        from nnsight.modeling.tp import UnshardableCheckpoint, check_tp_request, max_tp_size

        for config in (self._shardable(), self._config(plan=None, num_attention_heads=12)):
            limit = max_tp_size(config)
            for degree in range(2, 10):
                allowed = limit is not None and limit % degree == 0
                try:
                    check_tp_request(config, degree)
                except UnshardableCheckpoint:
                    assert not allowed, f"refused tp_size={degree} though the limit is {limit}"
                else:
                    assert allowed, f"allowed tp_size={degree} though the limit is {limit}"


class TestRequestedTpSize:
    """Reading the degree off whatever transformers would accept."""

    def test_no_config_asks_for_nothing(self):
        from nnsight.modeling.tp import requested_tp_size

        assert requested_tp_size(None) is None

    def test_a_degree_of_one_is_not_a_request(self):
        # tp_size=1 is the ordinary single-process case, not a split.
        from nnsight.modeling.tp import requested_tp_size

        assert requested_tp_size(type("D", (), {"tp_size": 1})()) is None

    def test_a_dataclass_and_a_dict_read_the_same(self):
        # transformers takes either, so the check has to see either.
        from nnsight.modeling.tp import requested_tp_size

        assert requested_tp_size(type("D", (), {"tp_size": 4})()) == 4
        assert requested_tp_size({"tp_size": 4}) == 4


class TestEveryLoadPathChecks:
    """The check has to sit on every `_load`, not only the base's.

    `TransformersModel` builds its model through `transformers.pipeline` and
    never calls `super()._load`, so a guard written once in `HuggingFaceModel`
    is dead for the class the tensor-parallel server actually loads through.
    That is exactly how this was first written, and a forced gpt2 deployment
    came up "tensor-parallel" across two cards regardless.
    """

    #: Loads that do not go through transformers, so a transformers tensor-parallel
    #: degree cannot reach them and cannot silently replicate. `DiffusionModel`
    #: builds a `diffusers` pipeline, which has no `base_model_tp_plan` and no
    #: `distributed_config` to honour.
    NOT_TRANSFORMERS = {"DiffusionModel"}

    def _overrides(self):
        import inspect

        # Imported explicitly: a subclass only exists once its module is, so a
        # sweep over `__subclasses__` otherwise passes by finding nothing --
        # which is the failure mode a coverage test can least afford.
        import nnsight.modeling.diffusion  # noqa: F401
        import nnsight.modeling.transformers  # noqa: F401
        from nnsight.modeling.huggingface import HuggingFaceModel

        subclasses, seen = [HuggingFaceModel], set()
        while subclasses:
            cls = subclasses.pop()
            if cls in seen:
                continue
            seen.add(cls)
            subclasses.extend(cls.__subclasses__())
        return {
            cls: inspect.getsource(cls.__dict__["_load"])
            for cls in seen
            # Shipped classes only. `__subclasses__` also sees every test double
            # any other module has defined, and a stub that never loads weights is
            # not something this guard has anything to say about.
            if cls.__module__.startswith("nnsight.")
            and "_load" in cls.__dict__
            and cls.__name__ not in self.NOT_TRANSFORMERS
        }

    def test_the_sweep_finds_the_overrides(self):
        # Guards the guard: if `_overrides` ever returns nothing, the assertion
        # below passes while checking nothing at all.
        names = {cls.__name__ for cls in self._overrides()}

        assert {"HuggingFaceModel", "TransformersModel"} <= names, names

    def test_every_load_override_refuses_an_impossible_degree(self):
        # Either it calls the shared check itself, or it delegates to a `_load`
        # that does. Anything else fetches the weights unchecked.
        unguarded = [
            cls.__name__
            for cls, source in self._overrides().items()
            if "_refuse_impossible_tp" not in source and "super()._load" not in source
        ]

        assert not unguarded, (
            f"these _load overrides never check the tensor-parallel degree: "
            f"{unguarded}. transformers will not check it either."
        )

    def test_the_class_the_server_loads_through_is_covered(self):
        # Stated separately from the sweep above because it is the specific one
        # that was missed, and a sweep can be quietly narrowed.
        import inspect

        from nnsight.modeling.transformers import TransformersModel

        assert "_refuse_impossible_tp" in inspect.getsource(TransformersModel._load)

    def test_a_load_with_no_degree_reads_no_config(self):
        # The check must cost the ordinary single-GPU path nothing: it runs on
        # every load, and a config read per load would be a real regression.
        from nnsight.modeling.huggingface import HuggingFaceModel

        read = []

        class Probe(HuggingFaceModel):
            def __init__(self):  # bypass Envoy construction; only the check matters
                self.revision = None

        import transformers

        original = transformers.AutoConfig.from_pretrained
        transformers.AutoConfig.from_pretrained = lambda *a, **k: read.append(a) or original(*a, **k)
        try:
            Probe()._refuse_impossible_tp("openai-community/gpt2", {})
            Probe()._refuse_impossible_tp("openai-community/gpt2", {"distributed_config": None})
        finally:
            transformers.AutoConfig.from_pretrained = original

        assert read == [], "the no-tensor-parallel path read a config it did not need"

    def _probe(self):
        from nnsight.modeling.huggingface import HuggingFaceModel

        class Probe(HuggingFaceModel):
            def __init__(self):  # bypass Envoy construction; only the check matters
                self.revision = None

        return Probe()

    def test_a_bare_tp_plan_auto_counts_as_a_request(self, monkeypatch):
        # ``tp_plan="auto"`` names no degree of its own — transformers shards
        # over whatever ranks the launcher provided — so the world size is the
        # degree being asked for. gpt2 publishes no plan, which makes a two-rank
        # ask the silent full-copy-per-rank case the check exists to refuse.
        from nnsight.modeling.tp import UnshardableCheckpoint

        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(UnshardableCheckpoint):
            self._probe()._refuse_impossible_tp(
                "openai-community/gpt2", {"tp_plan": "auto"}
            )

    def test_a_lone_rank_asking_auto_asks_for_nothing(self, monkeypatch):
        # With one rank there is nothing to shard over, so there is no degree to
        # check — and no config read, same as the ordinary path.
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        self._probe()._refuse_impossible_tp(
            "openai-community/gpt2", {"tp_plan": "auto"}
        )

    def test_a_custom_plan_dict_is_not_degree_checked(self, monkeypatch):
        # A dict plan overrides the checkpoint's published plan, so the published
        # plan's limit — which is all the check can read — says nothing about it.
        monkeypatch.setenv("WORLD_SIZE", "2")
        self._probe()._refuse_impossible_tp(
            "openai-community/gpt2", {"tp_plan": {"h.*.mlp.c_fc": "colwise"}}
        )

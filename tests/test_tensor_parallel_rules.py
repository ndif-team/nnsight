"""The tensor-parallel rule table, checked without a GPU.

`tests/test_transformers_tensor_parallel.py` proves the gather is *correct*, but
it needs two GPUs and so does not run everywhere. These tests cover the part that
goes wrong quietly and can be checked anywhere: whether the table still describes
the transformers it is running against, and whether the cases it refuses actually
raise.

The failure they exist for is version drift. transformers owns the list of
parallel styles; nnsight's table has to keep up with it, and a style that appears
upstream without a rule here is only discovered when someone deploys a model that
uses it.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from nnsight.intervention.interleaver import Interleaver
from nnsight.modeling.tp import (
    SHARDED_SIDES,
    UNSUPPORTED,
    TPFragments,
    UnsupportedParallelStyle,
)


def _upstream_styles() -> set:
    from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES

    return set(ALL_PARALLEL_STYLES._global_mapping)


class _FakeMesh:
    def __init__(self, size: int = 2) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeEnvoy:
    """The two things `instrument` reads off an envoy."""

    def __init__(self, module: torch.nn.Module, path: str = "model.thing") -> None:
        self._module = module
        self.path = path


def _module(style: str | None, **attrs) -> torch.nn.Module:
    module = torch.nn.Identity()
    if style is not None:
        module._hf_tp_plan = style
        module._hf_device_mesh = _FakeMesh()
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


class TestStyleCoverage:
    """Every style transformers knows about has a rule here, and vice versa."""

    def test_no_upstream_style_is_unaccounted_for(self):
        # The one that matters: a new style upstream with no rule here is a model
        # nnsight will refuse at deploy time. Better to find out at test time.
        missing = _upstream_styles() - (set(SHARDED_SIDES) | set(UNSUPPORTED))
        assert not missing, (
            f"transformers has parallel styles this version has no rule for: "
            f"{sorted(missing)}. Add each to SHARDED_SIDES (with the sides that "
            f"carry a shard) or to UNSUPPORTED."
        )

    def test_no_rule_names_a_style_that_no_longer_exists(self):
        stale = (set(SHARDED_SIDES) | set(UNSUPPORTED)) - _upstream_styles()
        assert not stale, (
            f"rules name parallel styles transformers no longer has: "
            f"{sorted(stale)} — probably renamed upstream."
        )

    def test_a_style_is_either_handled_or_refused_never_both(self):
        assert not (set(SHARDED_SIDES) & set(UNSUPPORTED))

    @pytest.mark.parametrize("sides", SHARDED_SIDES.values())
    def test_sides_are_input_or_output(self, sides):
        assert set(sides) <= {"input", "output"}


class TestInstrument:
    """What the interleaver does as the Envoy tree is built."""

    def test_starts_inert(self):
        interleaver = TPFragments()
        assert not interleaver.enabled
        assert not interleaver.tp_rules

    def test_an_unsharded_module_records_nothing(self, monkeypatch):
        interleaver = TPFragments()

        interleaver.instrument(_FakeEnvoy(_module(None)))

        assert not interleaver.enabled
        assert not interleaver.tp_rules

    def test_a_sharded_module_records_its_side_and_enables(self, monkeypatch):
        interleaver = TPFragments()

        interleaver.instrument(_FakeEnvoy(_module("colwise"), path="model.q_proj"))

        assert interleaver.enabled
        assert "model.q_proj.output" in interleaver.tp_rules

    def test_a_row_parallel_module_records_its_input(self, monkeypatch):
        interleaver = TPFragments()

        interleaver.instrument(_FakeEnvoy(_module("rowwise"), path="model.o_proj"))

        assert "model.o_proj.input" in interleaver.tp_rules
        assert "model.o_proj.output" not in interleaver.tp_rules

    @pytest.mark.parametrize("style", sorted(UNSUPPORTED))
    def test_a_refused_style_raises(self, style, monkeypatch):
        interleaver = TPFragments()

        with pytest.raises(UnsupportedParallelStyle, match=style):
            interleaver.instrument(_FakeEnvoy(_module(style)))

    def test_an_unknown_style_raises(self, monkeypatch):
        interleaver = TPFragments()

        with pytest.raises(UnsupportedParallelStyle, match="not a parallel style"):
            interleaver.instrument(_FakeEnvoy(_module("something_new_upstream")))


class TestSourceWarns:
    """`.source` reads under a sharded model warn and hand the value over.

    They cannot be gathered — the split axis moves through the forward — but
    plenty of them are whole anyway (anything past the layer that all-reduces),
    so refusing them all would block correct work. See
    `TPFragments.read`.
    """

    def _sharded(self, monkeypatch):
        fragments = TPFragments()
        fragments.instrument(
            _FakeEnvoy(_module("colwise"), path="model.layers.0.mlp.gate_proj")
        )
        return fragments

    def test_a_source_read_warns(self, monkeypatch):
        fragments = self._sharded(monkeypatch)

        with pytest.warns(UserWarning, match="split across ranks"):
            fragments.read("model.layers.0.mlp.source.self_gate_proj_0.output")

    def test_a_source_location_is_not_treated_as_a_fragment(self, monkeypatch):
        # The warning exists *because* it cannot be gathered: which axis a
        # `.source` value is split on changes through the forward.
        fragments = self._sharded(monkeypatch)

        assert not fragments.fragmented(
            "model.layers.0.mlp.source.self_gate_proj_0.output"
        )

    def test_it_warns_once_per_location_per_run(self, monkeypatch):
        # A read inside a generation loop fires every token; one caveat is a
        # caveat, hundreds are noise.
        fragments = self._sharded(monkeypatch)
        location = "model.layers.0.mlp.source.self_gate_proj_0.output"

        with pytest.warns(UserWarning):
            fragments.read(location)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fragments.read(location)
        assert not caught

    def test_a_new_run_warns_again(self, monkeypatch):
        # A long-lived model actor serves request after request; the second
        # user deserves the same caveat as the first.
        fragments = self._sharded(monkeypatch)
        location = "model.layers.0.mlp.source.self_gate_proj_0.output"

        with pytest.warns(UserWarning):
            fragments.read(location)
        fragments.begin()  # what Interleaver.__enter__ does at the start of a run
        with pytest.warns(UserWarning):
            fragments.read(location)

    def test_a_plain_module_read_does_not_warn(self, monkeypatch):
        fragments = self._sharded(monkeypatch)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fragments.read("model.layers.0.mlp.output")
        assert not caught

    def test_an_unsharded_model_is_never_asked(self, monkeypatch):
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

        # An expert-parallel model would fail at load; it must not be *placed*
        # as though it could be split.
        assert max_tp_size(self._config(
            plan={"layers.*.mlp.experts": "grouped_gemm"},
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
    """Sharding is refused on a transformers that shards incorrectly.

    Caught on a live deployment: the container ran 5.14.1 and a tied-embedding
    model came back with logits four times the vocabulary width, while the argmax
    — and so every eyeball check — stayed right.
    """

    def _check(self, monkeypatch, version: str):
        import transformers

        from nnsight.modeling.tp import fragments as tp_fragments

        monkeypatch.setattr(transformers, "__version__", version)
        monkeypatch.setattr(tp_fragments, "_version_checked", False)
        tp_fragments._check_transformers_version()

    def test_an_older_transformers_is_refused(self, monkeypatch):
        from nnsight.modeling.tp import UnsupportedTransformersVersion

        with pytest.raises(UnsupportedTransformersVersion, match="tie_word_embeddings"):
            self._check(monkeypatch, "5.14.1")

    @pytest.mark.parametrize(
        "version", ["5.15.0", "5.15.1", "6.0.0", "5.15.0.dev0", "5.16.0rc1"]
    )
    def test_the_floor_and_above_are_allowed(self, monkeypatch, version):
        # Including pre-releases of the fixed series: an editable transformers
        # checkout reports 5.15.0.dev0, which a plain >= would reject.
        self._check(monkeypatch, version)

    def test_it_only_runs_once(self, monkeypatch):
        # instrument() calls this per sharded module -- hundreds on a real model
        # -- so the import and version parse have to happen once, not per call.
        import transformers

        from nnsight.modeling.tp import fragments as tp_fragments

        monkeypatch.setattr(transformers, "__version__", "5.15.0")
        monkeypatch.setattr(tp_fragments, "_version_checked", False)
        tp_fragments._check_transformers_version()
        monkeypatch.setattr(transformers, "__version__", "0.1.0")
        tp_fragments._check_transformers_version()  # would raise if re-checked


class TestTheTwoGatesAgree:
    """Placement and load must refuse the same set of models.

    `max_tp_size` decides whether a model is *placed* tensor-parallel;
    `instrument` decides whether it can be *traced* that way. They ran different
    predicates -- one checked UNSUPPORTED, the other checked SHARDED_SIDES -- so a
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
        from nnsight.modeling.tp import SHARDED_SIDES, UNSUPPORTED, max_tp_size

        # Llama-4's actual plan. transformers does not register it in
        # ALL_PARALLEL_STYLES either, so the drift test below cannot see it.
        assert "colwise_rep" not in SHARDED_SIDES
        assert "colwise_rep" not in UNSUPPORTED
        assert max_tp_size(self._config({"layer.q_proj": "colwise_rep"})) is None

    def test_a_refused_style_is_still_refused(self):
        from nnsight.modeling.tp import max_tp_size

        assert max_tp_size(self._config({"layer.experts": "grouped_gemm"})) is None

    def test_a_known_style_still_places(self):
        from nnsight.modeling.tp import max_tp_size

        assert max_tp_size(self._config({"layer.q_proj": "colwise"})) == 8

    def test_every_placeable_plan_is_instrumentable(self):
        # The property, stated directly: anything max_tp_size accepts, instrument
        # must not raise on. Both now read SHARDED_SIDES, so this holds by
        # construction -- it is here to fail if they ever diverge again.
        from nnsight.modeling.tp import SHARDED_SIDES, max_tp_size

        for style in SHARDED_SIDES:
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

    def test_neither_side_is_gathered(self):
        from nnsight.modeling.tp import SHARDED_SIDES

        assert SHARDED_SIDES["moe_tp_experts"] == ()

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

    def test_a_style_that_really_slices_by_expert_is_still_refused(self):
        from nnsight.modeling.tp import UNSUPPORTED

        assert "grouped_gemm" in UNSUPPORTED
        assert "mla_kv_a_proj" in UNSUPPORTED


class TestEmbeddingColwiseIsWhole:
    """`embedding_colwise` all-reduces its output despite the name.

    `EmbeddingParallel._prepare_output_fn` ends in an unconditional
    `all_reduce_forward`; the `embedding_dim_sharding == 0` branch above it guards
    only the vocab masking. Gathering it again would hand users a tensor tp_size
    times too wide with a plausible first copy -- the tied-LM-head bug that
    MINIMUM_TRANSFORMERS exists for, reintroduced by this table.
    """

    def test_its_output_is_not_gathered(self):
        from nnsight.modeling.tp import SHARDED_SIDES

        assert SHARDED_SIDES["embedding_colwise"] == ()


class FakeMesh:
    """Just enough device mesh for the pure-tensor helpers."""

    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class TestReshardGuard:
    """What goes back to the model must be guarded like what came out of it.

    `_reshard`'s output is consumed by the model's own forward, so splitting a
    value that was never a fragment hands every rank a slice of something whole.
    Divisibility alone cannot tell the two apart: an integer `position_ids` of
    width 8 at tp=4 divides perfectly, and the result is wrong answers on every
    rank identically -- not a hang, and nothing to notice.
    """

    def _split_calls(self, monkeypatch):
        import transformers.integrations.tensor_parallel as tp

        seen = []

        def fake_split(tensor, mesh):
            seen.append(tensor)
            return tensor

        monkeypatch.setattr(tp, "split", fake_split)
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
        from nnsight.modeling.tp.fragments import _gather, _reshard

        gather_seen = self._split_calls(monkeypatch)
        import transformers.integrations.tensor_parallel as tp

        monkeypatch.setattr(tp, "all_gather", lambda t, m: gather_seen.append(t) or t)
        mesh = FakeMesh(4)

        _gather(tensor, mesh)
        assert not gather_seen, f"{why} should not have been gathered"

        split_seen = self._split_calls(monkeypatch)
        _reshard(tensor, mesh)
        assert not split_seen, f"{why} was split though it was never gathered"

    def test_a_real_shard_still_round_trips(self, monkeypatch):
        from nnsight.modeling.tp.fragments import _reshard

        split_seen = self._split_calls(monkeypatch)
        _reshard(torch.randn(1, 4, 8), FakeMesh(4))
        assert len(split_seen) == 1


class TestGatherShapeCheck:
    """A rule that names a whole value is caught the first time it fires.

    The rules are a claim about a transformers version, and most were settled by
    reading its source. This is what makes a wrong one loud: a side listed as
    sharded must actually widen by `world_size` when gathered. A value the model
    already made whole doesn't -- which is exactly the failure MINIMUM_TRANSFORMERS
    exists for, caught here for every version rather than one known one.
    """

    def _fragments(self, monkeypatch, gathered):
        import transformers.integrations.tensor_parallel as tp

        from nnsight.modeling.tp import TPFragments

        monkeypatch.setattr(tp, "all_gather", lambda tensor, mesh: gathered)
        fragments = TPFragments()
        fragments.enabled = True
        return fragments

    def test_a_value_that_did_not_widen_is_refused(self, monkeypatch):
        from nnsight.modeling.tp import UnsupportedParallelStyle

        mesh = FakeMesh(4)
        # The model's own hook already made it whole; gathering returns the same
        # width. Left unchecked, the user gets 4x the real tensor.
        value = torch.randn(1, 4, 2048)
        fragments = self._fragments(monkeypatch, torch.randn(1, 4, 2048))

        with pytest.raises(UnsupportedParallelStyle, match="times too wide|expected"):
            fragments._gather_whole("model.layer.output", value, mesh)

    def test_a_real_shard_passes(self, monkeypatch):
        mesh = FakeMesh(4)
        value = torch.randn(1, 4, 2048)
        fragments = self._fragments(monkeypatch, torch.randn(1, 4, 8192))

        whole = fragments._gather_whole("model.layer.output", value, mesh)
        assert whole.shape[-1] == 8192

    def test_it_checks_once_per_location(self, monkeypatch):
        # A generation loop revisits a location hundreds of times; the rule is a
        # property of the model, not of the visit.
        mesh = FakeMesh(4)
        fragments = self._fragments(monkeypatch, torch.randn(1, 4, 8192))
        fragments._gather_whole("model.layer.output", torch.randn(1, 4, 2048), mesh)

        import transformers.integrations.tensor_parallel as tp

        monkeypatch.setattr(tp, "all_gather", lambda t, m: torch.randn(1, 4, 2048))
        # Would raise if re-checked; the second visit is not checked.
        fragments._gather_whole("model.layer.output", torch.randn(1, 4, 2048), mesh)

    def test_an_ambiguous_value_is_not_judged(self, monkeypatch):
        # Two float tensors of different widths: nothing to compare, so the check
        # abstains rather than guessing which one the rule meant.
        mesh = FakeMesh(4)
        fragments = self._fragments(monkeypatch, torch.randn(1, 4, 2048))
        value = (torch.randn(1, 4, 2048), torch.randn(1, 4, 512))

        fragments._gather_whole("model.layer.output", value, mesh)


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

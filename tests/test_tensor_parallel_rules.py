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
    TPInterleaver,
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
        interleaver = TPInterleaver()
        assert not interleaver.enabled
        assert not interleaver.tp_rules

    def test_an_unsharded_module_records_nothing(self, monkeypatch):
        interleaver = TPInterleaver()
        monkeypatch.setattr(Interleaver, "instrument", lambda self, envoy: None)

        interleaver.instrument(_FakeEnvoy(_module(None)))

        assert not interleaver.enabled
        assert not interleaver.tp_rules

    def test_a_sharded_module_records_its_side_and_enables(self, monkeypatch):
        interleaver = TPInterleaver()
        monkeypatch.setattr(Interleaver, "instrument", lambda self, envoy: None)

        interleaver.instrument(_FakeEnvoy(_module("colwise"), path="model.q_proj"))

        assert interleaver.enabled
        assert "model.q_proj.output" in interleaver.tp_rules

    def test_a_row_parallel_module_records_its_input(self, monkeypatch):
        interleaver = TPInterleaver()
        monkeypatch.setattr(Interleaver, "instrument", lambda self, envoy: None)

        interleaver.instrument(_FakeEnvoy(_module("rowwise"), path="model.o_proj"))

        assert "model.o_proj.input" in interleaver.tp_rules
        assert "model.o_proj.output" not in interleaver.tp_rules

    @pytest.mark.parametrize("style", sorted(UNSUPPORTED))
    def test_a_refused_style_raises(self, style, monkeypatch):
        interleaver = TPInterleaver()
        monkeypatch.setattr(Interleaver, "instrument", lambda self, envoy: None)

        with pytest.raises(UnsupportedParallelStyle, match=style):
            interleaver.instrument(_FakeEnvoy(_module(style)))

    def test_an_unknown_style_raises(self, monkeypatch):
        interleaver = TPInterleaver()
        monkeypatch.setattr(Interleaver, "instrument", lambda self, envoy: None)

        with pytest.raises(UnsupportedParallelStyle, match="not a parallel style"):
            interleaver.instrument(_FakeEnvoy(_module("something_new_upstream")))


class TestSourceWarns:
    """`.source` reads under a sharded model warn and hand the value over.

    They cannot be gathered — the split axis moves through the forward — but
    plenty of them are whole anyway (anything past the layer that all-reduces),
    so refusing them all would block correct work. See
    `TPInterleaver._warn_source`.
    """

    def _sharded(self, monkeypatch):
        interleaver = TPInterleaver()
        monkeypatch.setattr(Interleaver, "instrument", lambda self, envoy: None)
        monkeypatch.setattr(Interleaver, "handle", lambda self, provider, value: value)
        interleaver.instrument(
            _FakeEnvoy(_module("colwise"), path="model.layers.0.mlp.gate_proj")
        )
        return interleaver

    def test_a_source_read_warns_and_returns_the_value(self, monkeypatch):
        interleaver = self._sharded(monkeypatch)
        monkeypatch.setattr(TPInterleaver, "observed", lambda self, provider: True)

        with pytest.warns(UserWarning, match="split across ranks"):
            value = interleaver.handle(
                "model.layers.0.mlp.source.self_gate_proj_0.output", 7
            )
        assert value == 7

    def test_it_warns_once_per_location_per_run(self, monkeypatch):
        # A read inside a generation loop fires every token; one caveat is a
        # caveat, hundreds are noise.
        interleaver = self._sharded(monkeypatch)
        monkeypatch.setattr(TPInterleaver, "observed", lambda self, provider: True)
        location = "model.layers.0.mlp.source.self_gate_proj_0.output"

        with pytest.warns(UserWarning):
            interleaver.handle(location, 7)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            interleaver.handle(location, 7)
        assert not caught

    def test_a_new_run_warns_again(self, monkeypatch):
        # A long-lived model actor serves request after request; the second
        # user deserves the same caveat as the first.
        interleaver = self._sharded(monkeypatch)
        monkeypatch.setattr(TPInterleaver, "observed", lambda self, provider: True)
        location = "model.layers.0.mlp.source.self_gate_proj_0.output"

        with pytest.warns(UserWarning):
            interleaver.handle(location, 7)
        interleaver._warned.clear()  # what __enter__ does at the start of a run
        with pytest.warns(UserWarning):
            interleaver.handle(location, 7)

    def test_a_plain_module_read_does_not_warn(self, monkeypatch):
        interleaver = self._sharded(monkeypatch)
        monkeypatch.setattr(TPInterleaver, "observed", lambda self, provider: True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert interleaver.handle("model.layers.0.mlp.output", 7) == 7
        assert not caught

    def test_an_unsharded_model_never_warns(self, monkeypatch):
        interleaver = TPInterleaver()
        monkeypatch.setattr(Interleaver, "handle", lambda self, provider, value: value)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert interleaver.handle("model.mlp.source.self_gate_proj_0.output", 7) == 7
        assert not caught


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

        from nnsight.modeling.tp import interleaver

        monkeypatch.setattr(transformers, "__version__", version)
        monkeypatch.setattr(interleaver, "_version_checked", False)
        interleaver._check_transformers_version()

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

        from nnsight.modeling.tp import interleaver

        monkeypatch.setattr(transformers, "__version__", "5.15.0")
        monkeypatch.setattr(interleaver, "_version_checked", False)
        interleaver._check_transformers_version()
        monkeypatch.setattr(transformers, "__version__", "0.1.0")
        interleaver._check_transformers_version()  # would raise if re-checked


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
        from nnsight.modeling.tp.interleaver import _gather, _reshard

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
        from nnsight.modeling.tp.interleaver import _reshard

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

    def _interleaver(self, monkeypatch, gathered):
        import transformers.integrations.tensor_parallel as tp

        from nnsight.modeling.tp import TPInterleaver

        monkeypatch.setattr(tp, "all_gather", lambda tensor, mesh: gathered)
        interleaver = TPInterleaver()
        interleaver.enabled = True
        # Stand in for the mediator machinery: the check runs before any of it.
        monkeypatch.setattr(type(interleaver), "observed", lambda self, p: True)
        return interleaver

    def test_a_value_that_did_not_widen_is_refused(self, monkeypatch):
        from nnsight.modeling.tp import UnsupportedParallelStyle

        mesh = FakeMesh(4)
        # The model's own hook already made it whole; gathering returns the same
        # width. Left unchecked, the user gets 4x the real tensor.
        value = torch.randn(1, 4, 2048)
        interleaver = self._interleaver(monkeypatch, torch.randn(1, 4, 2048))

        with pytest.raises(UnsupportedParallelStyle, match="times too wide|expected"):
            interleaver._gather_whole("model.layer.output", value, mesh)

    def test_a_real_shard_passes(self, monkeypatch):
        mesh = FakeMesh(4)
        value = torch.randn(1, 4, 2048)
        interleaver = self._interleaver(monkeypatch, torch.randn(1, 4, 8192))

        whole = interleaver._gather_whole("model.layer.output", value, mesh)
        assert whole.shape[-1] == 8192

    def test_it_checks_once_per_location(self, monkeypatch):
        # A generation loop revisits a location hundreds of times; the rule is a
        # property of the model, not of the visit.
        mesh = FakeMesh(4)
        interleaver = self._interleaver(monkeypatch, torch.randn(1, 4, 8192))
        interleaver._gather_whole("model.layer.output", torch.randn(1, 4, 2048), mesh)

        import transformers.integrations.tensor_parallel as tp

        monkeypatch.setattr(tp, "all_gather", lambda t, m: torch.randn(1, 4, 2048))
        # Would raise if re-checked; the second visit is not checked.
        interleaver._gather_whole("model.layer.output", torch.randn(1, 4, 2048), mesh)

    def test_an_ambiguous_value_is_not_judged(self, monkeypatch):
        # Two float tensors of different widths: nothing to compare, so the check
        # abstains rather than guessing which one the rule meant.
        mesh = FakeMesh(4)
        interleaver = self._interleaver(monkeypatch, torch.randn(1, 4, 2048))
        value = (torch.randn(1, 4, 2048), torch.randn(1, 4, 512))

        interleaver._gather_whole("model.layer.output", value, mesh)

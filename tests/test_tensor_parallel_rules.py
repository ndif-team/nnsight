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

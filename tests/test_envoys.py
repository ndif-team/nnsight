"""Tests for the ``envoys`` parameter / class attribute.

Covers:
- Default (None): children wrapped with base Envoy.
- Single class: all descendants wrapped with that class.
- Dict mapping: matched module classes wrapped with the mapped Envoy subclass;
  others fall back to Envoy. Matching walks the module's MRO.
- Propagation to grandchildren.
- ``NNsight`` subclass-level ``envoys`` attribute is used as a default.
- User-supplied ``envoys=`` kwarg overrides the class-level default.
"""

from collections import OrderedDict

import pytest
import torch

from nnsight import NNsight
from nnsight.intervention.envoy import Envoy


class _MyEnvoy(Envoy):
    pass


class _OtherEnvoy(Envoy):
    pass


class _LinearEnvoy(Envoy):
    pass


def _nested_net() -> torch.nn.Module:
    """Two-level module with a nested Sequential to test propagation."""
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("layer1", torch.nn.Linear(5, 10)),
                (
                    "inner",
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                ("act", torch.nn.ReLU()),
                                ("layer2", torch.nn.Linear(10, 2)),
                            ]
                        )
                    ),
                ),
            ]
        )
    )


class TestEnvoysParam:
    """Direct ``envoys=`` kwarg behavior on the base Envoy."""

    def test_default_is_none(self):
        model = NNsight(_nested_net())
        # All descendants (not the root) should be plain Envoy.
        for envoy in model.modules():
            if envoy is model:
                continue
            assert type(envoy) is Envoy, envoy.path

    def test_single_class_applies_to_all(self):
        model = NNsight(_nested_net(), envoys=_MyEnvoy)
        for envoy in model.modules():
            if envoy is model:
                continue
            assert type(envoy) is _MyEnvoy, envoy.path

    def test_dict_matches_and_falls_back(self):
        model = NNsight(_nested_net(), envoys={torch.nn.Linear: _LinearEnvoy})

        # layer1 and inner.layer2 are Linear -> _LinearEnvoy
        assert type(model.layer1) is _LinearEnvoy
        assert type(model.inner.layer2) is _LinearEnvoy
        # ReLU and the outer/inner Sequentials fall back to Envoy
        assert type(model.inner.act) is Envoy
        assert type(model.inner) is Envoy

    def test_dict_matches_subclass_via_mro(self):
        class MyLinear(torch.nn.Linear):
            pass

        net = torch.nn.Sequential(MyLinear(5, 10), torch.nn.ReLU())
        model = NNsight(net, envoys={torch.nn.Linear: _LinearEnvoy})
        assert type(model[0]) is _LinearEnvoy
        assert type(model[1]) is Envoy

    def test_propagates_to_grandchildren(self):
        model = NNsight(_nested_net(), envoys=_MyEnvoy)
        # The inner Sequential's own children should also be wrapped
        assert type(model.inner.act) is _MyEnvoy
        assert type(model.inner.layer2) is _MyEnvoy


class TestNNsightClassAttribute:
    """``NNsight`` subclasses can set a class-level ``envoys`` default."""

    def test_base_nnsight_default_is_none(self):
        assert NNsight.envoys is None

    def test_subclass_class_attribute_applied(self):
        class MyModel(NNsight):
            envoys = _MyEnvoy

        model = MyModel(_nested_net())
        for envoy in model.modules():
            if envoy is model:
                continue
            assert type(envoy) is _MyEnvoy, envoy.path

    def test_subclass_dict_class_attribute_applied(self):
        class MyModel(NNsight):
            envoys = {torch.nn.Linear: _LinearEnvoy}

        model = MyModel(_nested_net())
        assert type(model.layer1) is _LinearEnvoy
        assert type(model.inner.layer2) is _LinearEnvoy
        assert type(model.inner.act) is Envoy

    def test_user_kwarg_overrides_class_default(self):
        class MyModel(NNsight):
            envoys = _MyEnvoy

        model = MyModel(_nested_net(), envoys=_OtherEnvoy)
        for envoy in model.modules():
            if envoy is model:
                continue
            assert type(envoy) is _OtherEnvoy, envoy.path

    def test_user_kwarg_none_overrides_class_default(self):
        class MyModel(NNsight):
            envoys = _MyEnvoy

        model = MyModel(_nested_net(), envoys=None)
        for envoy in model.modules():
            if envoy is model:
                continue
            assert type(envoy) is Envoy, envoy.path

    def test_subclass_method_available_on_wrapped_children(self):
        class SpyEnvoy(Envoy):
            def marker(self):
                return f"spy:{self.path}"

        class MyModel(NNsight):
            envoys = {torch.nn.Linear: SpyEnvoy}

        model = MyModel(_nested_net())
        assert model.layer1.marker() == "spy:model.layer1"
        assert model.inner.layer2.marker() == "spy:model.inner.layer2"

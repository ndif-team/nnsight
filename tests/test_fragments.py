"""The seam a distributed runtime hangs its gather on.

`Fragments` says *what* is a piece of a larger value and how to reassemble it;
`Interleaver.handle` decides *when*. These pin the second half — the part every
runtime shares and none of them should have to reimplement — using a recording
stub rather than a real sharded model, so the ordering and the gating are
checkable without GPUs.

The real collectives are covered by `tests/tp` (transformers) and `tests/vllm`
(vLLM), both of which need several GPUs.
"""

from __future__ import annotations

import pytest
import torch

from nnsight import NNsight
from nnsight.intervention.fragments import Fragments
from nnsight.intervention.interleaver import Interleaver


class Recorder(Fragments):
    """Fragments that record what they were asked, and fake a 2x gather."""

    def __init__(self, locations=(), enabled=True):
        self.enabled = enabled
        self.locations = set(locations)
        self.calls: list[tuple[str, str]] = []
        self.instrumented: list[str] = []

    def instrument(self, envoy):
        self.instrumented.append(envoy.path)

    def fragmented(self, location):
        return location in self.locations

    def whole(self, location, value):
        self.calls.append(("whole", location))
        return torch.cat([value, value], dim=-1)

    def fragment(self, location, whole):
        self.calls.append(("fragment", location))
        return whole[..., : whole.shape[-1] // 2]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NNsight(torch.nn.Linear(4, 4))


def _traced(model, fragments, read: bool):
    model.interleaver.fragments = fragments
    captured = None
    with model.trace(torch.randn(1, 4)):
        if read:
            captured = model.output.save()
    return captured


class TestTheBracket:
    """A fragment is made whole before workers, and put back after."""

    def test_whole_then_fragment_in_that_order(self, model):
        fragments = Recorder(locations={"model.output"})
        _traced(model, fragments, read=True)

        kinds = [kind for kind, _ in fragments.calls if kind in ("whole", "fragment")]
        assert kinds == ["whole", "fragment"]

    def test_a_worker_sees_the_whole_value(self, model):
        fragments = Recorder(locations={"model.output"})
        captured = _traced(model, fragments, read=True)

        # The stub doubles the last dim; a worker reading the location gets that.
        assert captured.shape[-1] == 8

    def test_the_model_carries_on_with_a_fragment(self, model):
        fragments = Recorder(locations={"model.output"})
        _traced(model, fragments, read=True)

        assert ("fragment", "model.output") in fragments.calls


class TestGating:
    """Nothing happens unless something is actually waiting for the value."""

    def test_an_unread_location_is_not_gathered(self, model):
        fragments = Recorder(locations={"model.output"})
        _traced(model, fragments, read=False)

        assert not [c for c in fragments.calls if c[0] == "whole"]

    def test_a_disabled_runtime_is_never_asked(self, model):
        fragments = Recorder(locations={"model.output"}, enabled=False)
        _traced(model, fragments, read=True)

        assert fragments.calls == []

    def test_a_location_that_is_not_a_fragment_is_left_alone(self, model):
        fragments = Recorder(locations=set())
        captured = _traced(model, fragments, read=True)

        assert not [c for c in fragments.calls if c[0] == "whole"]
        assert captured.shape[-1] == 4

    def test_targeted_cache_gathers_its_subscription(self, model):
        fragments = Recorder(locations={"model.output"})
        model.interleaver.fragments = fragments
        with model.trace(torch.randn(1, 4)) as tracer:
            cache = tracer.cache(modules=[model])

        assert ("whole", "model.output") in fragments.calls
        assert cache["model"].output.shape[-1] == 8


class TestLifecycle:
    def test_the_tree_is_walked_past_instrument(self, model):
        fragments = Recorder()
        interleaver = Interleaver(fragments=fragments)
        NNsight(torch.nn.Linear(4, 4), interleaver=interleaver)

        assert fragments.instrumented, "no envoy reached the fragments"


class TestTheDefault:
    """The base is the identity, so an ordinary model pays nothing."""

    def test_nothing_is_a_fragment(self):
        assert Fragments().fragmented("anything") is False

    def test_it_starts_disabled(self):
        assert Fragments().enabled is False

    def test_an_ordinary_model_has_none_at_all(self):
        assert NNsight(torch.nn.Linear(4, 4)).interleaver.fragments is None

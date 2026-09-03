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

from functools import partial

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
        return torch.cat([value, value], dim=-1), partial(self._undo, location)

    def _undo(self, location, whole):
        self.calls.append(("fragment", location))
        return whole[..., : whole.shape[-1] // 2]

    def split(self, location, whole):
        # A value that was never gathered: same cut, no gather to reverse.
        self.calls.append(("split", location))
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


class TestASkippedValueIsNormalized:
    """A `.skip` replacement never came out of the module.

    It is the caller's own whole value, while everything `handle` does assumes
    this device's *piece*. `handle_replacement` cuts it down to a piece first, so
    the rest of the path treats it exactly like one the module produced. Without
    that a row-parallel skip was all-reduced on the way out and every rank read
    back ``world_size`` times what it wrote, with nothing raising.

    Asserted on what a reader sees, not on which methods ran: the point is that a
    skip round-trips, however that is arranged.
    """

    def test_a_reader_sees_what_was_written(self, model):
        fragments = Recorder(locations={"model.output"})
        model.interleaver.fragments = fragments

        with model.trace(torch.randn(1, 4)):
            model.skip(torch.ones(1, 4))
            seen = model.output.save()

        # 4 is what was written. 8 would be the Recorder's doubling — the
        # replacement assembled as though it were one device's piece.
        assert seen.shape[-1] == 4
        assert torch.equal(seen, torch.ones(1, 4))
        assert ("split", "model.output") in fragments.calls

    def test_it_is_cut_down_even_with_nobody_reading(self, model):
        """The model's forward carries on with it whether or not anyone looked."""
        fragments = Recorder(locations={"model.output"})
        model.interleaver.fragments = fragments

        with model.trace(torch.randn(1, 4)):
            model.skip(torch.ones(1, 4))

        assert ("split", "model.output") in fragments.calls
        # Nothing was parked, so nothing was assembled either.
        assert ("whole", "model.output") not in fragments.calls


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

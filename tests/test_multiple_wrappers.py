"""Wrapping one module in more than one :class:`NNsight` at a time.

Each wrapper builds its own envoy tree and its own interleaver over the *same*
underlying module, so the wrappers must stay independent: a trace or an
intervention through one leaves the others seeing the unmodified model, and every
wrapper keeps working no matter how many share the module.
"""

import gc
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

import nnsight
from nnsight import NNsight

INPUT_SIZE = 5
HIDDEN = 10
OUTPUT_SIZE = 2


@pytest.fixture
def model():
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("layer1", torch.nn.Linear(INPUT_SIZE, HIDDEN)),
                ("layer2", torch.nn.Linear(HIDDEN, OUTPUT_SIZE)),
            ]
        )
    )


class _Sourceable(nn.Module):
    """A module with a source-instrumentable forward (a named op, ``torch_relu_0``)
    and a skippable submodule (``fc``), for the source/skip wrapper tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(INPUT_SIZE, INPUT_SIZE)

    def forward(self, x):
        h = self.fc(x)
        return torch.relu(h)


@pytest.fixture
def x():
    return torch.rand((1, INPUT_SIZE))


class TestMultipleWrappers:
    @torch.no_grad()
    def test_two_wrappers_both_work(self, model, x):
        first, second = NNsight(model), NNsight(model)
        with first.trace(x):
            out1 = first.layer2.output.save()
        with second.trace(x):
            out2 = second.layer2.output.save()
        assert torch.allclose(out1, out2)

    @torch.no_grad()
    def test_modifications_are_independent(self, model, x):
        first, second = NNsight(model), NNsight(model)
        with second.trace(x):
            baseline = second.layer2.output.clone().save()
        with first.trace(x):
            first.layer1.output[:] = 0
            modified = first.layer2.output.save()
        with second.trace(x):
            unaffected = second.layer2.output.save()
        assert not torch.allclose(modified, baseline)  # the intervention landed
        assert torch.allclose(unaffected, baseline)  # only on its own wrapper

    @torch.no_grad()
    def test_interleaved_reuse(self, model, x):
        first, second = NNsight(model), NNsight(model)
        with first.trace(x):
            first_a = first.layer2.output.save()
        with second.trace(x):
            other = second.layer2.output.save()
        with first.trace(x):
            first_b = first.layer2.output.save()
        assert torch.allclose(first_a, other)
        assert torch.allclose(first_a, first_b)

    @torch.no_grad()
    def test_three_wrappers(self, model, x):
        outs = []
        for _ in range(3):
            wrapper = NNsight(model)
            with wrapper.trace(x):
                out = wrapper.layer2.output.save()
            outs.append(out)
        assert torch.allclose(outs[0], outs[1])
        assert torch.allclose(outs[1], outs[2])

    @torch.no_grad()
    def test_new_wrapper_after_modification(self, model, x):
        first = NNsight(model)
        with first.trace(x):
            first.layer1.output[:] = 1.0
            modified = first.layer2.output.save()
        second = NNsight(model)  # made after the modified trace
        with second.trace(x):
            fresh = second.layer2.output.save()
        assert not torch.allclose(modified, fresh)

    @torch.no_grad()
    def test_skip_with_multiple_wrappers(self, model, x):
        first, second = NNsight(model), NNsight(model)
        with first.trace(x):
            first.layer2.output = first.layer1.output[:, :OUTPUT_SIZE]
            skipped = first.output.save()
        with second.trace(x):
            normal = second.output.save()
        assert not torch.allclose(skipped, normal)

    @torch.no_grad()
    def test_save_intermediate_layers(self, model, x):
        first, second = NNsight(model), NNsight(model)
        with first.trace(x):
            a1 = first.layer1.output.save()
            a2 = first.layer2.output.save()
        with second.trace(x):
            b1 = second.layer1.output.save()
            b2 = second.layer2.output.save()
        assert torch.allclose(a1, b1)
        assert torch.allclose(a2, b2)

    def test_gradients_with_multiple_wrappers(self, model, x):
        def grad(wrapper):
            with wrapper.trace(x):
                activation = wrapper.layer1.output
                loss = wrapper.output.sum()
                with loss.backward():
                    captured = activation.grad.clone().save()
            return captured

        assert torch.allclose(grad(NNsight(model)), grad(NNsight(model)))


class TestManyWrappers:
    @torch.no_grad()
    def test_many_wrappers_stay_consistent(self, model, x):
        reference = None
        wrappers = [NNsight(model) for _ in range(10)]
        for wrapper in wrappers:
            with wrapper.trace(x):
                out = wrapper.layer2.output.save()
            if reference is None:
                reference = out.clone()
            else:
                assert torch.allclose(out, reference)

    @torch.no_grad()
    def test_still_works_after_wrappers_are_collected(self, model, x):
        for _ in range(5):
            wrapper = NNsight(model)
            with wrapper.trace(x):
                wrapper.layer2.output.save()
        gc.collect()
        fresh = NNsight(model)
        with fresh.trace(x):
            out = fresh.layer2.output.save()
        assert isinstance(out, torch.Tensor)


class TestMultipleWrapperSourceSkip:
    """Source and skip go through the module's single installed controller, which
    routes to whichever wrapper's trace is running — so they, too, stay independent
    across wrappers sharing a module."""

    @torch.no_grad()
    def test_source_matches_across_wrappers(self, x):
        net = _Sourceable()
        first, second = NNsight(net), NNsight(net)
        with first.trace(x):
            a = first.source.torch_relu_0.output.save()
        with second.trace(x):
            b = second.source.torch_relu_0.output.save()
        assert torch.allclose(a, b)

    @torch.no_grad()
    def test_source_intervention_is_per_wrapper(self, x):
        net = _Sourceable()
        first, second = NNsight(net), NNsight(net)
        with second.trace(x):
            baseline = second.output.clone().save()
        with first.trace(x):
            first.source.torch_relu_0.output[:] = 0  # kill the relu output via first
            zeroed = first.output.save()
        with second.trace(x):
            unaffected = second.output.save()
        assert (zeroed == 0).all()  # the source edit landed on `first`
        assert torch.allclose(unaffected, baseline)  # and only on `first`

    @torch.no_grad()
    def test_skip_is_per_wrapper(self, x):
        net = _Sourceable()
        first, second = NNsight(net), NNsight(net)
        with first.trace(x):
            first.fc.skip(torch.zeros(1, INPUT_SIZE))  # fc doesn't run -> relu(0)=0
            skipped = first.output.save()
        with second.trace(x):
            normal = second.output.save()
        assert (skipped == 0).all()  # first skipped fc
        assert (normal != 0).any()  # second ran it

    @torch.no_grad()
    def test_forward_survives_after_sourcing_wrapper_is_gone(self, x):
        # The controller is installed on the module permanently; once the wrapper
        # that sourced it is gone, a later plain forward must still run — the source
        # state keys on the (surviving) interleaver, not the collected envoy. This is
        # the local analogue of the remote cross-request contamination.
        net = _Sourceable()
        wrapper = NNsight(net)
        with wrapper.trace(x):
            wrapper.source.torch_relu_0.output.save()
        del wrapper
        gc.collect()
        out = net(x)  # the controller runs with no active trace
        assert out.shape == (1, INPUT_SIZE)

"""Tests for single-device intervention values on a model split across two GPUs.

`Tower` passes two tensors across its four stages: a running ``stream`` and a
``shelf`` of earlier streams. When `Tower` is split across ``cuda:0`` and
``cuda:1``, ``stages[3]`` receives ``stream`` on ``cuda:1`` and ``shelf`` on
``cuda:0``, and `Tower.forward` reads both the ``stream`` that entered
``stages[3]`` (``entering``) and ``shelf`` again in ``readout``. These tests
verify that values served to intervention code are placed on one device and
that in-place edits, replacements, weight accesses, ad-hoc calls, skips, and
gradient reads match a single-GPU reference model with identical weights.

Requires two CUDA devices; skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

import nnsight
from nnsight import NNsight

accelerate = pytest.importorskip("accelerate")
pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")

WIDTH = 12


class Stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blend = torch.nn.Linear(WIDTH, WIDTH)

    def forward(self, stream, shelf):
        return stream + torch.tanh(self.blend(stream + shelf.mean(-2)))


class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = torch.nn.ModuleList(Stage() for _ in range(4))
        self.readout = torch.nn.Linear(WIDTH, 5)

    def forward(self, entry):
        stream = entry
        shelf = entry.new_zeros(*entry.shape[:-1], 1, WIDTH)
        for index, stage in enumerate(self.stages):
            entering = stream
            grown = stage(stream, shelf)
            if index % 2 == 0:  # append the stage's input stream to `shelf` at even stages
                shelf = torch.cat([shelf.to(stream.device), stream.unsqueeze(-2)], dim=-2)
            stream = grown
        # `entering` (the stream passed to `stages[3]`) and `shelf` are read again here.
        return self.readout(stream + entering + shelf.mean(-2).to(stream.device))


def towers(trainable=False):
    """Return two `NNsight`-wrapped `Tower` models with identical weights: one
    on ``cuda:0``, and one split across ``cuda:0`` and ``cuda:1``."""
    torch.manual_seed(0)
    whole = Tower().requires_grad_(trainable).to("cuda:0")
    split = Tower().requires_grad_(trainable)
    split.load_state_dict(whole.state_dict())
    accelerate.dispatch_model(
        split,
        device_map={"stages.0": 0, "stages.1": 0, "stages.2": 1, "stages.3": 1, "readout": 1},
    )
    return NNsight(whole), NNsight(split)


@pytest.fixture
def entry():
    torch.manual_seed(1)
    return torch.randn(2, 3, WIDTH)


def both(block, entry, trainable=False):
    """Run ``block(model, tracer)`` on the single-device and split models;
    return the saved CPU results."""
    results = []
    for model in towers(trainable):
        with model.trace(entry) as tracer:
            kept = nnsight.save(block(model, tracer).cpu())
        results.append(kept)
    return results


def test_a_stage_inputs_arrive_on_one_device(entry):
    # stages[3] receives `stream` from cuda:1 and `shelf` from cuda:0.
    def block(model, tracer):
        stream, shelf = model.stages[3].inputs[0]
        assert stream.device == shelf.device
        return torch.cat([shelf, stream.unsqueeze(-2)], dim=-2)

    whole, split = both(block, entry)
    assert torch.allclose(split, whole)


def test_an_in_place_edit_reaches_what_the_forward_reads_later(entry):
    # Edit both inputs of stages[3] in place. Tower.forward reads both tensors
    # again in readout after stages[3] returns; stream is on cuda:1, so the
    # in-place edit on its cuda:0 copy must be copied back into the cuda:1 tensor.
    def block(model, tracer):
        stream, shelf = model.stages[3].inputs[0]
        stream += 1.5
        shelf[..., 0] = 4.0
        return model.output

    whole, split = both(block, entry)
    assert torch.allclose(split, whole, atol=1e-6)
    unedited = both(lambda model, tracer: model.output, entry)[1]
    assert not torch.allclose(split, unedited, atol=1e-3)


def test_an_assigned_value_takes_the_original_place(entry):
    def block(model, tracer):
        model.stages[2].output = model.stages[2].output * 0.5
        return model.output

    whole, split = both(block, entry)
    assert torch.allclose(split, whole, atol=1e-6)


def test_a_weight_can_be_combined_with_a_value_from_another_device(entry):
    # readout.weight is on cuda:1; the stream entering stages[1] is on cuda:0.
    def block(model, tracer):
        stream = model.stages[1].inputs[0][0]
        stream += model.readout.weight[2]
        return model.output

    whole, split = both(block, entry)
    assert torch.allclose(split, whole, atol=1e-6)


def test_a_gradient_started_at_a_moved_input(entry):
    # The stream entering stages[3] is on cuda:1 and is served as a copy on
    # cuda:0. Marking the copy with requires_grad_(True) copies it back into
    # the cuda:1 input tensor so autograd propagates gradients to the copy.
    def block(model, tracer):
        stream = model.stages[3].inputs[0][0]
        stream.requires_grad_(True)
        with model.output.sum().backward():
            grad = stream.grad.clone()
        return grad

    whole, split = both(block, entry)
    assert torch.allclose(split, whole, atol=1e-6)


def test_a_read_leaves_a_trainable_models_graph_alone(entry):
    # With trainable weights, activations already require grad. Reading
    # stages[3]'s input copy must not modify the original tensor in place or
    # disturb the autograd graph.
    def block(model, tracer):
        model.stages[3].inputs[0][0].sum()
        model.output.sum().backward()
        return model.stages[0].blend.weight.grad

    whole, split = both(block, entry, trainable=True)
    assert torch.allclose(split, whole, atol=1e-6)


def test_a_gradient_read_on_a_trainable_model(entry):
    # On a trainable model, the copy served on cuda:0 is not used by the
    # forward pass; reading stream.grad in a backward trace serves the gradient
    # of the original cuda:1 tensor on cuda:0.
    def block(model, tracer):
        stream = model.stages[3].inputs[0][0]
        with model.output.sum().backward():
            grad = stream.grad.clone()
        return grad

    whole, split = both(block, entry, trainable=True)
    assert torch.allclose(split, whole, atol=1e-6)


def test_one_tensor_read_at_two_locations_has_one_gradient_and_it_can_be_edited(entry):
    # stages[2].output and stages[3].inputs[0][0] are two copies of the same
    # cuda:1 tensor. Zeroing the gradient on one copy in place zeroes the
    # gradient flowing to earlier stages.
    def block(model, tracer):
        early = model.stages[1].output
        out = model.stages[2].output
        stream = model.stages[3].inputs[0][0]
        with model.output.sum().backward():
            seen = stream.grad.clone()
            out.grad[:] = 0
            upstream = early.grad.clone()
        return torch.cat([seen.flatten(), upstream.flatten()])

    whole, split = both(block, entry, trainable=True)
    assert torch.allclose(split, whole, atol=1e-6)


def test_an_in_place_edit_under_inference_mode(entry):
    def block(model, tracer):
        model.stages[3].inputs[0][0][..., :4] += 1.5
        return model.output

    with torch.inference_mode():
        whole, split = both(block, entry)
        unedited = both(lambda model, tracer: model.output, entry)[1]
    assert torch.allclose(split, whole, atol=1e-6)
    assert not torch.allclose(split, unedited, atol=1e-3)


def test_an_edit_followed_by_an_assignment_keeps_both(entry):
    # Edit `shelf` in place before reassigning `stages[3].inputs` with
    # `shelf * 0.5`; `Tower.forward` reads the in-place edited `shelf` in
    # `readout`, while `stages[3]` receives `shelf * 0.5`.
    def block(model, tracer):
        stream, shelf = model.stages[3].inputs[0]
        shelf[..., 0] = 4.0
        model.stages[3].inputs = ((stream, shelf * 0.5), {})
        return model.output

    whole, split = both(block, entry)
    assert torch.allclose(split, whole, atol=1e-6)


def test_a_trainable_weight_and_a_module_call_come_back_on_the_same_device(entry):
    def block(model, tracer):
        stream = model.stages[1].inputs[0][0]
        lens = model.readout(stream)  # readout is on cuda:1
        return lens + stream[..., :5] + model.readout.weight[2, :5].detach()

    whole, split = both(block, entry, trainable=True)
    assert torch.allclose(split, whole, atol=1e-6)


def test_a_setting_changed_beside_a_moved_tensor_is_kept(entry):
    # Move `shelf` from positional `args` to `kwargs` so the returned `inputs`
    # tuple has a different structure than the served one.
    def block(model, tracer):
        (stream, shelf), _ = model.stages[3].inputs
        model.stages[3].inputs = ((stream,), {"shelf": shelf * 0.5})
        return model.output

    whole, split = both(block, entry)
    assert torch.allclose(split, whole, atol=1e-6)


def test_a_skipped_module_and_an_ad_hoc_call_on_the_other_device(entry):
    def block(model, tracer):
        stream = model.stages[3].inputs[0][0]
        model.stages[3].blend.skip(model.stages[3].blend(stream) * 0.5)
        return model.output

    whole, split = both(block, entry)
    assert torch.allclose(split, whole, atol=1e-6)


def test_a_model_on_one_device_has_no_device_view(entry):
    whole, _ = towers()
    with whole.trace(entry):
        seen = nnsight.save([whole.interleaver.devices])
        whole.stages[0].output
    assert seen == [None]

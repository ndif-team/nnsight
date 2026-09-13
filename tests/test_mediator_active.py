"""Pause/resume intervention hooks while another request uses the same model."""

from types import SimpleNamespace

import pytest
import torch

from nnsight import NNsight
from nnsight.intervention.hooks import (
    operation_fn_hook,
    operation_input_hook,
    operation_output_hook,
)
from nnsight.intervention.interleaver import Cancelation, Interleaver, Mediator


class ScheduledCalls(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()
        self._scheduler_interleaver = None

    def forward(self, value):
        mediators = list(self._scheduler_interleaver.mediators)
        outputs = []
        for step in range(1, 5):
            for mediator in mediators:
                mediator.active = step % 2 == 0
            outputs.append(self.layer(value * step))
        for mediator in mediators:
            mediator.active = True
        return torch.cat(outputs)


@pytest.fixture
def scheduled_model():
    module = ScheduledCalls()
    model = NNsight(module)
    module._scheduler_interleaver = model.interleaver
    return model


@pytest.mark.parametrize("access", ["input", "output"])
def test_pending_hook_survives_pause_and_resumes(scheduled_model, access):
    with scheduled_model.trace(torch.ones(1, 1)):
        if access == "input":
            scheduled_model.layer.input[:] = 20
        else:
            scheduled_model.layer.output[:] = 20
        result = scheduled_model.output.save()

    # The first call belongs to another request. Only the next active call
    # consumes the pending intervention; later calls remain untouched.
    assert result[:, 0].tolist() == [1, 20, 3, 4]


def test_iteration_counts_only_active_calls(scheduled_model):
    with scheduled_model.trace(torch.ones(1, 1)) as tracer:
        captured = [].save()
        for step in tracer.iter[:2]:
            captured.append(scheduled_model.layer.output.item())

    assert captured == [2, 4]


def test_cache_records_only_active_calls(scheduled_model):
    with scheduled_model.trace(torch.ones(1, 1)) as tracer:
        cache = tracer.cache(modules=[scheduled_model.layer], include_inputs=True).save()

    entries = cache[scheduled_model.layer.path]
    assert [entry.output.item() for entry in entries] == [2, 4]
    assert [entry.input.item() for entry in entries] == [2, 4]


def test_intervention_inherits_inference_mode_for_inplace_edits():
    model = NNsight(torch.nn.Sequential(torch.nn.Identity()))
    with torch.inference_mode():
        value = torch.ones(1, 1)
        with model.trace(value):
            model[0].output[:] = 7
            result = model.output.save()
    assert result.item() == 7


@pytest.mark.parametrize("preserve_locals", [False, True])
def test_cancelled_intervention_preserves_locals_for_remote_collection(preserve_locals):
    frame = SimpleNamespace(f_locals={})
    info = SimpleNamespace(frame=frame)

    def intervention(mediator, info):
        try:
            captured = ["prefill", "decode"]
            mediator.request("next_execution")
        except Cancelation as exc:
            mediator.exception(exc)

    mediator = Mediator(intervention, info=info)
    interleaver = Interleaver(mediators=[mediator])
    with interleaver:
        mediator.cancel(preserve_locals=preserve_locals)
    assert not mediator.alive
    if preserve_locals:
        assert frame.f_locals["captured"] == ["prefill", "decode"]
    else:
        assert frame.f_locals == {}


@pytest.mark.parametrize(
    "register, hooks_name, suffix",
    [
        (operation_input_hook, "pre_hooks", ".input.i0"),
        (operation_output_hook, "post_hooks", ".output.i0"),
        (operation_fn_hook, "fn_hooks", ".fn"),
    ],
)
def test_source_hook_stays_pending_during_pause(register, hooks_name, suffix):
    mediator = Mediator(lambda: None, info=None)
    observed = []
    mediator.handle = lambda provider, value: observed.append((provider, value)) or value
    accessor = SimpleNamespace(path="layer.op", pre_hooks=[], post_hooks=[], fn_hooks=[])
    handle = register(mediator, accessor)
    callbacks = vars(accessor)[hooks_name]
    value = object()

    mediator.active = False
    assert callbacks[0](value) is value
    assert len(callbacks) == 1
    assert observed == []

    mediator.active = True
    assert callbacks[0](value) is value
    assert observed == [("layer.op" + suffix, value)]
    assert callbacks == []
    handle.remove()


def test_provided_values_do_not_advance_paused_mediator():
    mediator = Mediator(lambda: None, info=None)
    observed = []
    mediator.handle = lambda provider, value: observed.append((provider, value)) or value
    interleaver = Interleaver.__new__(Interleaver)
    interleaver.mediators = [mediator]

    mediator.active = False
    assert interleaver.handle("logits", "other request", iterate=True) == "other request"
    assert observed == []
    assert mediator.iteration_tracker["logits"] == 0

    mediator.active = True
    assert interleaver.handle("logits", "own request", iterate=True) == "own request"
    assert observed == [("logits.i0", "own request")]
    assert mediator.iteration_tracker["logits"] == 1


def test_mediator_serialization_preserves_pause():
    mediator = Mediator(lambda: None, info=None)
    assert mediator.active
    mediator.active = False
    state = mediator.__getstate__()
    restored = Mediator.__new__(Mediator)
    restored.__setstate__(state)
    assert not restored.active

    # Mediators serialized by older clients begin active as before.
    del state["active"]
    restored.__setstate__(state)
    assert restored.active


import pytest
import torch
import torch.nn as nn

import nnsight
from nnsight.intervention.editing import EditingTracer
from nnsight.intervention.envoy import Envoy


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Adapter(nn.Module):
    """An attachable module (adapter/LoRA/SAE-style) with an observable submodule."""

    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(8, 8)

    def forward(self, x):
        return self.inner(x)


class BatchedEnvoy(Envoy):
    """An envoy that combines invokes by stacking them on dim 0, so attachment tests
    can exercise per-invoke narrowing without a real model (the base Envoy passes a
    lone invoke through and refuses to batch several)."""

    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}


class BatchedEnvoy(Envoy):
    """An Envoy that stacks several invokes into one batch, so attachment tests can
    exercise per-invoke narrowing without loading a real batching model."""

    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}


class Repeat(nn.Module):
    """Calls one submodule repeatedly, so its location fires several times per
    forward — the same shape as a generation loop, without needing a real one."""

    STEPS = 3

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)

    def forward(self, x):
        for _ in range(self.STEPS):
            x = self.fc1(x)
        return x


@pytest.fixture
def x():
    return torch.randn(2, 8)


def normal(model, x):
    return model.fc2(torch.relu(model.fc1(x)))


def zeroed_fc1(model, x):
    # fc1 output forced to zeros -> relu(0)=0 -> fc2(0)
    return model.fc2(torch.relu(torch.zeros(2, 8)))


class TestEditing:
    def test_edit_replays_on_future_trace(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.edit() as (tracer, edited):
            edited.fc1.output[:] = 0
        captured = {}
        with edited.trace(x):
            captured["out"] = edited.output
        assert torch.allclose(captured["out"], zeroed_fc1(model, x))

    def test_original_unaffected_by_copy_edit(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.edit() as (tracer, edited):
            edited.fc1.output[:] = 0
        captured = {}
        with envoy.trace(x):
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], normal(model, x))
        assert envoy._edits == []
        assert len(edited._edits) == 1

    def test_edit_inplace(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.edit(inplace=True):
            envoy.fc1.output[:] = 0
        captured = {}
        with envoy.trace(x):
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], zeroed_fc1(model, x))
        assert len(envoy._edits) == 1

    def test_clear_edits(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.edit(inplace=True):
            envoy.fc1.output[:] = 0
        envoy.clear_edits()
        captured = {}
        with envoy.trace(x):
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], normal(model, x))

    def test_edit_persists_across_traces(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.edit(inplace=True):
            envoy.fc1.output[:] = 0
        captured = {}
        with envoy.trace(x):
            captured["a"] = envoy.output
        with envoy.trace(x):
            captured["b"] = envoy.output
        assert torch.allclose(captured["a"], captured["b"])
        assert torch.allclose(captured["a"], zeroed_fc1(model, x))

    def test_edits_stack(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.edit(inplace=True):
            envoy.fc1.output[:] = 0
        with envoy.edit(inplace=True):
            envoy.fc2.output[:] = 7
        captured = {}
        with envoy.trace(x):
            captured["out"] = envoy.output
        assert len(envoy._edits) == 2
        assert torch.allclose(captured["out"], torch.full((2, 8), 7.0))

    def test_edit_inplace_binds_tracer(self, x):
        # An edit block is a trace block, so entering it binds the tracer (and its
        # iter API). inplace=True has no copy to hand back, so that's all it binds.
        envoy = Envoy(MLP())
        with envoy.edit(inplace=True) as tracer:
            assert isinstance(tracer, EditingTracer)
            assert hasattr(tracer, "iter")
            envoy.fc1.output[:] = 0
        captured = {}
        with envoy.trace(x):
            captured["fc1"] = envoy.fc1.output
        assert torch.allclose(captured["fc1"], torch.zeros(2, 8))

    def test_edit_copy_binds_tracer_and_copy(self, x):
        # inplace=False stores the edit on a copy, so the copy is bound alongside
        # the tracer and the original is left clean.
        envoy = Envoy(MLP())
        with envoy.edit() as (tracer, edited):
            assert isinstance(tracer, EditingTracer)
            assert edited is not envoy
            edited.fc1.output[:] = 0
        assert envoy._edits == []
        assert len(edited._edits) == 1

    def test_edit_applies_before_trace_intervention(self, x):
        # An edit's swap is visible to a same-trace intervention that reads it.
        model = MLP()
        envoy = Envoy(model)
        with envoy.edit(inplace=True):
            envoy.fc1.output[:] = 0
        captured = {}
        with envoy.trace(x):
            captured["fc1"] = envoy.fc1.output
        assert torch.allclose(captured["fc1"], torch.zeros(2, 8))


class TestEditingWithAttachment:
    """An edit can attach a module to the tree and route activations through it with
    ``hook=True`` — the way adapters/LoRA/SAEs are applied. ``hook=True`` runs the
    attachment's full ``__call__`` (not just ``.forward``) so its own submodules'
    hooks fire and their activations become observable in every future trace, e.g.
    at ``.adapter.inner.output``. The attachment runs inside the edit's replay
    (a worker), and the observation lives in the trace body (a second worker), so
    this exercises a module call whose nested hooks serve another worker mid-call."""

    def test_attachment_output_replaces_activation(self, x):
        model = MLP()
        envoy = Envoy(model)
        envoy.fc1.adapter = Adapter()
        with envoy.edit(inplace=True):
            acts = envoy.fc1.output
            envoy.fc1.output[:] = envoy.fc1.adapter(acts, hook=True)
        with envoy.trace(x):
            captured = nnsight.save({})
            captured["inner"] = envoy.fc1.adapter.inner.output.clone()
            captured["fc1"] = envoy.fc1.output.clone()
        # fc1.output is now the adapter's output, which is its inner submodule's output.
        assert torch.allclose(captured["fc1"], captured["inner"])

    def test_attachment_internals_match_direct_forward(self, x):
        model = MLP()
        envoy = Envoy(model)
        envoy.fc1.adapter = Adapter()
        with envoy.edit(inplace=True):
            acts = envoy.fc1.output
            envoy.fc1.output[:] = envoy.fc1.adapter(acts, hook=True)
        with envoy.trace(x):
            captured = nnsight.save({})
            captured["inner"] = envoy.fc1.adapter.inner.output.clone()
        # The observed inner output equals running the real adapter on the raw fc1
        # output (recomputed here from the clean model).
        raw_fc1 = model.fc1(x)
        assert torch.allclose(captured["inner"], model.fc1.adapter.inner(raw_fc1))

    def test_attachment_changes_model_output(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            clean = envoy.output.save()
        envoy.fc1.adapter = Adapter()
        with envoy.edit(inplace=True):
            acts = envoy.fc1.output
            envoy.fc1.output[:] = envoy.fc1.adapter(acts, hook=True)
        with envoy.trace(x):
            edited = envoy.output.save()
        assert not torch.allclose(clean, edited)

    def test_attachment_applies_once_without_iter(self, x):
        # A plain edit block reads and swaps the location's *first* occurrence, so
        # the attachment is applied once even though the layer runs several times.
        # The loop is open because that is the claim under test: the attachment
        # fires once, and a loop naming a count the run doesn't reach is an error.
        envoy = Envoy(Repeat())
        envoy.fc1.adapter = Adapter()
        with envoy.edit(inplace=True):
            acts = envoy.fc1.output
            envoy.fc1.output[:] = envoy.fc1.adapter(acts, hook=True)
        with pytest.warns(UserWarning, match="never reached"):
            with envoy.trace(x) as tracer:
                seen = nnsight.save([])
                for _ in tracer.all():
                    seen.append(envoy.fc1.adapter.inner.output.clone())
        assert len(seen) == 1

    def test_attachment_applies_every_occurrence_with_iter(self, x):
        # Putting the passthrough under the edit tracer's iter re-applies it at every
        # occurrence — the way an adapter is applied on each step of a generation loop.
        envoy = Envoy(Repeat())
        envoy.fc1.adapter = Adapter()
        with envoy.edit(inplace=True) as tracer:
            for _ in tracer.iter[:]:
                acts = envoy.fc1.output
                envoy.fc1.output[:] = envoy.fc1.adapter(acts, hook=True)
        with envoy.trace(x) as tracer:
            seen = nnsight.save([])
            for _ in tracer.iter[: Repeat.STEPS]:
                seen.append(envoy.fc1.adapter.inner.output.clone())
        assert len(seen) == Repeat.STEPS
        # Each occurrence feeds on the previous one's result, so they differ.
        assert not torch.allclose(seen[0], seen[1])

    def test_attachment_narrows_per_invoke(self):
        # The edit runs the attachment once on the whole combined batch (an edit is a
        # whole-batch worker: batch_group None), and each invoke's worker narrows the
        # attachment's activations down to its own rows.
        model = MLP()
        envoy = BatchedEnvoy(model)
        envoy.fc1.adapter = Adapter()
        with envoy.edit(inplace=True):
            acts = envoy.fc1.output
            envoy.fc1.output[:] = envoy.fc1.adapter(acts, hook=True)
        xa, xb = torch.randn(2, 8), torch.randn(3, 8)
        with envoy.trace() as tracer:
            captured = nnsight.save({})
            with tracer.invoke(xa):
                captured["a"] = envoy.fc1.adapter.inner.output
            with tracer.invoke(xb):
                captured["b"] = envoy.fc1.adapter.inner.output
        # Each invoke sees only its own rows, holding its own prompt's values.
        assert captured["a"].shape[0] == 2 and captured["b"].shape[0] == 3
        assert torch.allclose(
            captured["a"], model.fc1.adapter.inner(model.fc1(xa)), atol=1e-5
        )
        assert torch.allclose(
            captured["b"], model.fc1.adapter.inner(model.fc1(xb)), atol=1e-5
        )

    def test_attachment_batched(self, x):
        # A batched input flows through the attachment with the batch dim intact.
        model = MLP()
        envoy = Envoy(model)
        envoy.fc1.adapter = Adapter()
        with envoy.edit(inplace=True):
            acts = envoy.fc1.output
            envoy.fc1.output[:] = envoy.fc1.adapter(acts, hook=True)
        with envoy.trace(torch.randn(4, 8)):
            captured = nnsight.save({})
            captured["inner"] = envoy.fc1.adapter.inner.output
        assert captured["inner"].shape[0] == 4

    def test_attachment_narrows_per_invoke(self):
        # The edit runs the attachment once over the whole combined batch (an edit is
        # a whole-batch worker), and each invoke reads back only its own rows.
        model = MLP()
        envoy = BatchedEnvoy(model)
        envoy.fc1.adapter = Adapter()
        with envoy.edit(inplace=True):
            acts = envoy.fc1.output
            envoy.fc1.output[:] = envoy.fc1.adapter(acts, hook=True)
        xa, xb = torch.randn(2, 8), torch.randn(3, 8)
        with envoy.trace() as tracer:
            captured = nnsight.save({})
            with tracer.invoke(xa):
                captured["a"] = envoy.fc1.adapter.inner.output
            with tracer.invoke(xb):
                captured["b"] = envoy.fc1.adapter.inner.output
        # Each invoke sees its own row count and its own values.
        inner = model.fc1.adapter.inner
        assert torch.allclose(captured["a"], inner(model.fc1(xa)), atol=1e-5)
        assert torch.allclose(captured["b"], inner(model.fc1(xb)), atol=1e-5)


class TestEditSerialization:
    """Edits ride with the model to a remote server: they live in envoy._edits,
    which serializes by value, so each edit Mediator must survive a round-trip.
    remote="local" exercises the whole path (serialize -> hide local modules ->
    deserialize -> run) in-process."""

    @torch.no_grad()
    def test_edit_survives_serialization(self, gpt2):
        with gpt2.edit() as (tracer, edited):  # non-inplace copy; gpt2 stays clean
            edited.transformer.h[0].output[0][:] = 0

        with edited.trace("The Eiffel Tower is in", remote="local"):
            out = edited.transformer.h[0].output[0].save()
        assert torch.all(out == 0)

    @torch.no_grad()
    def test_unedited_model_unaffected(self, gpt2):
        with gpt2.edit():  # a copy is edited; the original keeps no edits
            pass
        with gpt2.trace("The Eiffel Tower is in", remote="local"):
            out = gpt2.transformer.h[0].output[0].save()
        assert torch.any(out != 0)

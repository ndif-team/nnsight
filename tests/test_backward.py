
import pytest
import torch
import nnsight
import torch.nn as nn

from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import OutOfOrderError


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def x():
    return torch.randn(2, 8)


def reference_grad(model, x):
    """The gradient flowing into fc1's output, computed with plain autograd."""
    a1 = model.fc1(x)
    a1.retain_grad()
    model.fc2(torch.relu(a1)).sum().backward()
    return a1.grad


class TestBackward:
    def test_grad_matches_reference(self, x):
        torch.manual_seed(0)
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            a1 = envoy.fc1.output
            loss = envoy.output.sum()
            with loss.backward():
                captured["g"] = a1.grad.clone()
        assert torch.allclose(captured["g"], reference_grad(model, x))

    def test_grad_edit_is_visible(self, x):
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            a1 = envoy.fc1.output
            loss = envoy.output.sum()
            with loss.backward():
                captured["g1"] = a1.grad.clone()
                a1.grad = a1.grad * 2
                captured["g2"] = a1.grad.clone()
        assert torch.equal(captured["g2"], captured["g1"] * 2)

    def test_grad_edit_propagates_downstream(self, x):
        # Doubling fc1's output gradient doubles fc1's weight gradient.
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            a1 = envoy.fc1.output
            loss = envoy.output.sum()
            with loss.backward():
                a1.grad = a1.grad * 2
        edited = model.fc1.weight.grad.clone()

        model.zero_grad()
        a1 = model.fc1(x)
        model.fc2(torch.relu(a1)).sum().backward()
        reference = model.fc1.weight.grad

        assert torch.allclose(edited, 2 * reference)

    def test_multiple_grads_in_backward_order(self, x):
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            a1 = envoy.fc1.output
            a2 = envoy.fc2.output
            loss = envoy.output.sum()
            with loss.backward():
                # fc2's gradient flows first, then fc1's.
                captured["g2"] = a2.grad.clone()
                captured["g1"] = a1.grad.clone()
        assert captured["g1"].shape == (2, 8)
        assert captured["g2"].shape == (2, 8)

    def test_out_of_order_grad_raises(self, x):
        model = MLP()
        envoy = Envoy(model)
        with pytest.raises(OutOfOrderError):
            with envoy.trace(x):
                a1 = envoy.fc1.output
                a2 = envoy.fc2.output
                loss = envoy.output.sum()
                with loss.backward():
                    a1.grad  # fc1's gradient flows last...
                    a2.grad  # ...so requesting fc2's now is out of order

    def test_grad_escapes_via_variable(self, x):
        # Values assigned in the (nested) backward block reach the caller.
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            a1 = envoy.fc1.output
            loss = envoy.output.sum()
            with loss.backward():
                g = nnsight.save(a1.grad.clone())
        assert torch.allclose(g, reference_grad(model, x))


class TestBareBackward:
    def test_backward_without_with_block_is_untouched(self):
        t = torch.tensor([2.0], requires_grad=True)
        (t * 3).sum().backward()
        assert torch.equal(t.grad, torch.tensor([3.0]))


class _BatchEnvoy(Envoy):
    """An Envoy that stacks each invoke's tensor input into one batch, so batching
    behavior can be exercised without a real batching model."""

    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}


class TestBackwardAcrossInvokes:
    """Gradients when more than one invoke shares the forward pass.

    All invokes are combined into one batched forward, so they share one
    autograd graph. Each invoke's loss is built from its own rows, so its
    gradient must be its own rows too -- and the first ``.backward()`` frees
    the graph for every invoke after it.
    """

    @pytest.fixture
    def batched(self):
        torch.manual_seed(0)
        model = MLP()
        return model, _BatchEnvoy(model)

    @pytest.fixture
    def inputs(self):
        # Different row counts, so each invoke sits at a different offset in
        # the batch and a slice taken from the wrong place is visible.
        torch.manual_seed(1)
        return torch.randn(2, 8), torch.randn(3, 8)

    def test_each_invoke_gets_its_own_gradient(self, batched, inputs):
        model, envoy = batched
        x0, x1 = inputs
        expected = [reference_grad(model, x0), reference_grad(model, x1)]

        with envoy.trace() as tracer:
            with tracer.invoke(x0):
                a0 = envoy.fc1.output
                with envoy.output.sum().backward(retain_graph=True):
                    g0 = nnsight.save(a0.grad.clone())
            with tracer.invoke(x1):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward():
                    g1 = nnsight.save(a1.grad.clone())

        assert g0.shape == x0.shape
        assert g1.shape == x1.shape
        assert torch.allclose(g0, expected[0], atol=1e-6)
        assert torch.allclose(g1, expected[1], atol=1e-6)

    @pytest.mark.parametrize("target", [0, 1, 2])
    def test_gradient_from_any_invoke_of_three(self, target):
        torch.manual_seed(0)
        model = MLP()
        envoy = _BatchEnvoy(model)
        torch.manual_seed(2)
        xs = [torch.randn(2, 8), torch.randn(3, 8), torch.randn(1, 8)]
        expected = reference_grad(model, xs[target])

        captured = {}
        with envoy.trace() as tracer:
            for index, x in enumerate(xs):
                with tracer.invoke(x):
                    if index == target:
                        a1 = envoy.fc1.output
                        with envoy.output.sum().backward(retain_graph=True):
                            captured["g"] = nnsight.save(a1.grad.clone())
                    else:
                        nnsight.save(envoy.output.sum())

        got = captured["g"]
        assert got.shape == xs[target].shape
        assert torch.allclose(got, expected, atol=1e-6)

        # Negative control: not another invoke's rows. (Only comparable where
        # the row counts happen to match; shapes differ otherwise, which is
        # itself the check.)
        for index, x in enumerate(xs):
            if index != target and x.shape == got.shape:
                assert not torch.allclose(
                    got, reference_grad(model, x), atol=1e-4
                )

    def test_grad_edit_stays_within_its_invoke(self, batched, inputs):
        model, envoy = batched
        x0, x1 = inputs
        expected1 = reference_grad(model, x1)

        with envoy.trace() as tracer:
            with tracer.invoke(x0):
                a0 = envoy.fc1.output
                with envoy.output.sum().backward(retain_graph=True):
                    a0.grad = a0.grad * 3.0
                    g0 = nnsight.save(a0.grad.clone())
            with tracer.invoke(x1):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward():
                    g1 = nnsight.save(a1.grad.clone())

        assert torch.allclose(g0, reference_grad(model, x0) * 3.0, atol=1e-6)
        # The other invoke is untouched by the edit.
        assert torch.allclose(g1, expected1, atol=1e-6)


class TestFreedGraphDiagnostic:
    """Autograd's "second time" message, restated in terms of invokes."""

    @pytest.fixture
    def envoy(self):
        torch.manual_seed(0)
        return _BatchEnvoy(MLP())

    @staticmethod
    def _two_invokes(envoy, first_kwargs, second_kwargs):
        torch.manual_seed(3)
        x0, x1 = torch.randn(2, 8), torch.randn(3, 8)
        with envoy.trace() as tracer:
            with tracer.invoke(x0):
                a0 = envoy.fc1.output
                with envoy.output.sum().backward(**first_kwargs):
                    nnsight.save(a0.grad.clone())
            with tracer.invoke(x1):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward(**second_kwargs):
                    nnsight.save(a1.grad.clone())

    def test_message_names_the_cause_and_the_fix(self, envoy):
        with pytest.raises(RuntimeError) as info:
            self._two_invokes(envoy, {}, {})

        message = str(info.value)

        # The cause, in nnsight's terms rather than autograd's.
        assert "share one autograd graph" in message
        assert "single batched forward pass" in message
        # The fix, and torch's original text for anyone who wants it.
        assert "retain_graph=True" in message
        assert "Original error from torch.autograd" in message

    def test_message_notes_when_this_call_already_retains(self, envoy):
        # The failing call retains; it is the earlier one that did not. Saying
        # "pass retain_graph=True" alone would point at a flag already set.
        with pytest.raises(RuntimeError) as info:
            self._two_invokes(envoy, {}, {"retain_graph": True})

        assert "did not" in str(info.value)

    def test_other_runtime_errors_are_not_reworded(self, envoy):
        # Only the freed-graph message is translated.
        torch.manual_seed(4)
        x = torch.randn(2, 8)
        with pytest.raises(RuntimeError) as info:
            with envoy.trace(x):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward(gradient=torch.zeros(3, 3)):
                    nnsight.save(a1.grad.clone())

        assert "share one autograd graph" not in str(info.value)

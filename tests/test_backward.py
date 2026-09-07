
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
    behaviour can be exercised without a real batching model.

    Mirrors the helper in ``test_batching.py``; kept local because that file is
    deliberately all ``@torch.no_grad()`` and these are the gradient tests.
    """

    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}


class TestBatchedInvokeGradients:
    """An invoke's gradient is its own rows.

    Invokes share one batched forward, so a module's output is one tensor with
    every invoke's rows in it. Reading ``.grad`` inside an invoke has to narrow
    to that invoke's slice — reading the whole batch, or another invoke's slice,
    is a plausible-looking tensor that is silently the wrong gradient.

    Row counts differ per invoke on purpose, so a slice taken at the wrong
    offset fails on shape rather than on values that might round close.
    """

    @pytest.fixture
    def batched(self):
        torch.manual_seed(0)
        model = MLP()
        return model, _BatchEnvoy(model)

    @pytest.mark.parametrize("target", [0, 1, 2])
    def test_the_gradient_read_is_that_invokes_rows(self, batched, target):
        model, envoy = batched
        torch.manual_seed(1)
        xs = [torch.randn(2, 8), torch.randn(3, 8), torch.randn(1, 8)]
        expected = reference_grad(model, xs[target])

        captured = {}
        with envoy.trace() as tracer:
            for index, x in enumerate(xs):
                with tracer.invoke(x):
                    if index == target:
                        a1 = envoy.fc1.output
                        with envoy.output.sum().backward():
                            captured["grad"] = nnsight.save(a1.grad.clone())
                    else:
                        nnsight.save(envoy.output.sum())

        got = captured["grad"]

        # The batch is 6 rows; this invoke's is 2, 3 or 1 of them.
        assert got.shape == xs[target].shape
        assert torch.allclose(got, expected, atol=1e-6)

    def test_the_gradient_read_is_not_another_invokes_rows(self, batched):
        # Same row count on both invokes, so the shape check above cannot be
        # what catches a wrong offset -- only the values can.
        model, envoy = batched
        torch.manual_seed(2)
        first, second = torch.randn(3, 8), torch.randn(3, 8)
        expected = reference_grad(model, second)
        other = reference_grad(model, first)

        captured = {}
        with envoy.trace() as tracer:
            with tracer.invoke(first):
                nnsight.save(envoy.output.sum())
            with tracer.invoke(second):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward():
                    captured["grad"] = nnsight.save(a1.grad.clone())

        got = captured["grad"]

        assert torch.allclose(got, expected, atol=1e-6)
        assert not torch.allclose(got, other, atol=1e-4)

    def test_each_invoke_reads_its_own_gradient(self, batched):
        # Two invokes both taking a gradient. They share one autograd graph, so
        # every backward but the last needs retain_graph=True.
        model, envoy = batched
        torch.manual_seed(3)
        first, second = torch.randn(2, 8), torch.randn(3, 8)
        expected_first = reference_grad(model, first)
        expected_second = reference_grad(model, second)

        with envoy.trace() as tracer:
            with tracer.invoke(first):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward(retain_graph=True):
                    grad_first = nnsight.save(a1.grad.clone())
            with tracer.invoke(second):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward():
                    grad_second = nnsight.save(a1.grad.clone())

        assert grad_first.shape == first.shape
        assert grad_second.shape == second.shape
        assert torch.allclose(grad_first, expected_first, atol=1e-6)
        assert torch.allclose(grad_second, expected_second, atol=1e-6)

    def test_a_gradient_edit_stays_in_its_invoke(self, batched):
        model, envoy = batched
        torch.manual_seed(4)
        first, second = torch.randn(2, 8), torch.randn(3, 8)
        expected_first = reference_grad(model, first)
        expected_second = reference_grad(model, second)

        with envoy.trace() as tracer:
            with tracer.invoke(first):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward(retain_graph=True):
                    a1.grad = a1.grad * 3.0
                    grad_first = nnsight.save(a1.grad.clone())
            with tracer.invoke(second):
                a1 = envoy.fc1.output
                with envoy.output.sum().backward():
                    grad_second = nnsight.save(a1.grad.clone())

        assert torch.allclose(grad_first, expected_first * 3.0, atol=1e-6)
        # The other invoke's rows are untouched by the edit.
        assert torch.allclose(grad_second, expected_second, atol=1e-6)

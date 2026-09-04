
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

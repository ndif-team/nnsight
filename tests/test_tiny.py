"""
Tests for base NNsight functionality using a tiny two-layer model.

These tests cover core features that work with any PyTorch model:
- Basic tracing and saving
- Gradient access and modification
- Conditionals and iteration
- Session management
- Early stopping
"""

import pytest
import torch

from nnsight import NNsight


class TestBasicTracing:
    """Tests for basic tracing and value access."""

    @torch.no_grad()
    def test_save_output(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test that module outputs can be saved and accessed."""
        with tiny_model.trace(tiny_input):
            hs = tiny_model.layer2.output.save()

        assert isinstance(hs, torch.Tensor)

    def test_torch_creation_operations(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test that torch tensor creation operations work within trace."""
        with tiny_model.trace(tiny_input):
            l1_output = tiny_model.layer1.output
            torch.arange(l1_output.shape[0], l1_output.shape[1])
            torch.empty(l1_output.shape)
            torch.eye(l1_output.shape[0])
            torch.full(l1_output.shape, 5)
            torch.linspace(l1_output.shape[0], l1_output.shape[1], 5)
            torch.logspace(l1_output.shape[0], l1_output.shape[1], 5)
            torch.ones(l1_output.shape)
            torch.rand(l1_output.shape)
            torch.randint(5, l1_output.shape)
            torch.randn(l1_output.shape)
            torch.randperm(l1_output.shape[0])
            torch.zeros(l1_output.shape)


class TestGradients:
    """Tests for gradient access and modification."""

    def test_grad_access_and_modify(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test accessing and modifying gradients within backward context."""
        with tiny_model.trace(tiny_input):
            l1o = tiny_model.layer1.output
            loss = tiny_model.output.sum()

            with loss.backward():
                l1_grad = l1o.grad.clone().save()
                l1o.grad = l1o.grad.clone() * 2
                l1_grad_double = l1o.grad.save()

        assert torch.equal(l1_grad * 2, l1_grad_double)


class TestConditionals:
    """Tests for conditional logic within traces."""

    def test_true_conditional(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test that true conditionals execute their body."""
        with tiny_model.trace(tiny_input):
            num = 5
            if num > 0:
                tiny_model.layer1.output[:] = 1
                l1_out = tiny_model.layer1.output.save()

        assert isinstance(l1_out, torch.Tensor)
        assert torch.all(l1_out == 1).item()

    def test_false_conditional(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test that false conditionals skip their body."""
        with tiny_model.trace(tiny_input):
            num = 5
            if num < 0:
                tiny_model.layer1.output[:] = 1
                l1_out = tiny_model.layer1.output.save()

            l2_out = tiny_model.layer2.output.save()

        with pytest.raises(UnboundLocalError):
            l1_out

        assert isinstance(l2_out, torch.Tensor)

    def test_tensor_as_condition(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test using tensor boolean as condition."""
        with tiny_model.trace(tiny_input):
            out = tiny_model.layer1.output
            out[:, 0] = 1
            if out[:, 0] != 1:
                tiny_model.layer1.output[:] = 1
                l1_out = tiny_model.layer1.output.save()

        with pytest.raises(UnboundLocalError):
            l1_out

    def test_multiple_dependent_conditionals(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test multiple conditionals that depend on each other."""
        with tiny_model.trace(tiny_input):
            num = 5
            l1_out = tiny_model.layer1.output
            l2_out = tiny_model.layer2.output.save()
            if num > 0:
                l1_out[:] = 1

            if l1_out[:, 0] != 1:
                tiny_model.layer2.output[:] = 2

            if l1_out[:, 0] == 1:
                l2_out[:] = 3

        assert torch.all(l2_out == 3).item()

    def test_nested_conditionals(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test nested conditional logic."""
        with tiny_model.trace(tiny_input):
            num = 5
            if num > 0:  # True
                l1_out = tiny_model.layer1.output.save()

                if num > 0:  # True
                    tiny_model.layer1.output[:] = 1

                if num < 0:  # False
                    tiny_model.layer1.output[:] = 2

            if num < 0:  # False
                tiny_model.layer2.output[:] = 0

                if num > 0:  # True (but parent is false)
                    l2_out = tiny_model.layer2.output.save()

        assert isinstance(l1_out, torch.Tensor)
        assert torch.all(l1_out == 1).item()
        with pytest.raises(UnboundLocalError):
            l2_out


class TestSession:
    """Tests for session-based operations."""

    def test_cross_trace_intervention(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test using values from one trace in another."""
        with tiny_model.session():
            with tiny_model.trace(tiny_input):
                l1_out = tiny_model.layer1.output.save()

            with tiny_model.trace(tiny_input):
                l1_out[:, 2] = 5

        assert l1_out[:, 2] == 5

    def test_conditional_trace(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test conditionally creating traces."""
        with tiny_model.session():
            num = 5
            if num > 0:
                with tiny_model.trace(tiny_input):
                    output = tiny_model.output.save()

        assert isinstance(output, torch.Tensor)

    def test_conditional_iteration(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test conditionals within iteration."""
        with tiny_model.session():
            result = [].save()
            for item in [0, 1, 2]:
                if item % 2 == 0:
                    with tiny_model.trace(tiny_input):
                        result.append(item)

        assert result == [0, 2]

    def test_bridge_protocol(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test bridging values from session scope into traces."""
        with tiny_model.session():
            val = 0
            with tiny_model.trace(tiny_input):
                tiny_model.layer1.output[:] = val
                l1_out = tiny_model.layer1.output.save()

        assert torch.all(l1_out == 0).item()

    def test_loop_with_break(self, tiny_model: NNsight):
        """Test loop with break statement in session."""
        with tiny_model.session():
            l = [].save()
            l.append(0)

            for item in [1, 2, 3, 4]:
                if item == 3:
                    break
                l.append(item)
            l.append(5)

        assert l == [0, 1, 2, 5]

    def test_nested_iterator(self, tiny_model: NNsight):
        """Test nested iteration in session."""
        with tiny_model.session():
            l = [].save()
            l.append([0])
            l.append([1])
            l.append([2])
            l2 = [].save()
            for item in l:
                for item_2 in item:
                    l2.append(item_2)

        assert l2 == [0, 1, 2]

    def test_nnsight_builtins(self, tiny_model: NNsight):
        """Test that Python builtins work correctly in sessions."""
        with tiny_model.session():
            nn_list = [].save()
            sesh_list = [].save()
            apply_list = [].save()

            for l in [nn_list, sesh_list, apply_list]:
                l.append(int)
                l.append("Hello World")
                l.append({"a": "1"})

        assert nn_list == sesh_list
        assert sesh_list == apply_list


class TestEarlyStopping:
    """Tests for early termination of traces."""

    def test_stop_protocol(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test that tracer.stop() terminates execution."""
        with tiny_model.trace(tiny_input) as tracer:
            l1_out = tiny_model.layer1.output.save()
            tracer.stop()
            l2_out = tiny_model.layer2.output.save()

        assert isinstance(l1_out, torch.Tensor)

        with pytest.raises(UnboundLocalError):
            l2_out

    def test_stop_prevents_operations(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Test that operations after stop are not executed."""
        with tiny_model.trace(tiny_input) as tracer:
            l1_out = tiny_model.layer1.output
            tracer.stop()
            l1_out_double = (l1_out * 2).save()

        with pytest.raises(UnboundLocalError):
            l1_out_double

    def test_bridged_node_cleanup(self, tiny_model: NNsight):
        """Test that bridged nodes are properly cleaned up on break."""
        with tiny_model.session():
            l = [].save()
            for item in [0, 1, 2]:
                if item == 2:
                    break
                l.append(item)

        assert l == [0, 1]

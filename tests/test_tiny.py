from collections import OrderedDict

import pytest
import torch

from nnsight import NNsight

input_size = 5
hidden_dims = 10
output_size = 2


@pytest.fixture(scope="module")
def tiny_model(device: str):

    net = torch.nn.Sequential(
        OrderedDict(
            [
                ("layer1", torch.nn.Linear(input_size, hidden_dims)),
                ("layer2", torch.nn.Linear(hidden_dims, output_size)),
            ]
        )
    )

    return NNsight(net).to(device)


@pytest.fixture
def tiny_input():
    return torch.rand((1, input_size))


@torch.no_grad()
def test_tiny_1(tiny_model: NNsight, tiny_input: torch.Tensor):

    with tiny_model.trace(tiny_input):

        hs = tiny_model.layer2.output.save()

    assert isinstance(hs, torch.Tensor)


def test_grad_setting(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input):
        
        l1o = tiny_model.layer1.output

        loss = tiny_model.output.sum()

        with loss.backward():
            l1_grad = l1o.grad.clone().save()
            l1o.grad = l1o.grad.clone() * 2
            l1_grad_double = l1o.grad.save()

    assert torch.equal(l1_grad * 2, l1_grad_double)


def test_external_proxy_intervention_executed_locally(
    tiny_model: NNsight, tiny_input: torch.Tensor
):
    with tiny_model.session() as sesh:
        with tiny_model.trace(tiny_input) as tracer_1:
            l1_out = tiny_model.layer1.output.save()

        with tiny_model.trace(tiny_input) as tracer_2:
            l1_out[:, 2] = 5

    assert l1_out[:, 2] == 5


def test_early_stop_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input) as tracer:
        l1_out = tiny_model.layer1.output.save()
        tracer.stop()
        l2_out = tiny_model.layer2.output.save()

    assert isinstance(l1_out, torch.Tensor)

    with pytest.raises(UnboundLocalError):
        l2_out


def test_true_conditional_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input) as tracer:
        num = 5
        if num > 0:
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert torch.all(l1_out == 1).item()


def test_false_conditional_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input) as tracer:
        num = 5
        if num < 0:
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

        # check that the condition does not persist outside the context
        l2_out = tiny_model.layer2.output.save()

    with pytest.raises(UnboundLocalError):
        l1_out
        
    assert isinstance(l2_out, torch.Tensor)


def test_node_as_condition(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test a Tensor a boolean value as a result of a boolean operation on an InterventionProxy"""

    with tiny_model.trace(tiny_input) as tracer:
        out = tiny_model.layer1.output
        out[:, 0] = 1
        if out[:, 0] != 1:
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

    with pytest.raises(UnboundLocalError):
        l1_out


def test_multiple_dependent_conditionals(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that interventions defined within different Intervention contexts can be referenced if their conditions evaluated to True."""

    with tiny_model.trace(tiny_input) as tracer:
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


def test_nested_conditionals(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input) as tracer:
        num = 5
        if num > 0:  # True
            l1_out = tiny_model.layer1.output.save()

            if num > 0:  # True
                tiny_model.layer1.output[:] = 1

            if num < 0:  # False
                tiny_model.layer1.output[:] = 2

        if num < 0:  # False
            tiny_model.layer2.output[:] = 0

            if num > 0:  # True
                l2_out = tiny_model.layer2.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert torch.all(l1_out == 1).item()
    with pytest.raises(UnboundLocalError):
        l2_out

def test_conditional_trace(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as session:
        num = 5
        if num > 0:
            with tiny_model.trace(tiny_input):
                output = tiny_model.output.save()

    assert isinstance(output, torch.Tensor)


def test_conditional_iteration(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as session:
        result = [].save()
        for item in [0,1,2]:
            if item % 2 == 0:
                with tiny_model.trace(tiny_input):
                    result.append(item)

    assert result == [0, 2]


def test_bridge_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as session:
        val = 0
        with tiny_model.trace(tiny_input):
            tiny_model.layer1.output[:] = (
                val  # fetches the val proxy using the bridge protocol
            )
            l1_out = tiny_model.layer1.output.save()

    assert torch.all(l1_out == 0).item()


def test_sequential_graph_based_context_exit(tiny_model: NNsight):
    with tiny_model.session() as session:
        l = [].save()
        l.append(0)

        for item in [1, 2, 3, 4]:
            if item == 3:
                break
            l.append(item)
        l.append(5)
        

    assert l == [0, 1, 2, 5]


def test_tracer_stop(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input) as tracer:
        l1_out = tiny_model.layer1.output
        tracer.stop()
        l1_out_double = (l1_out * 2).save()

    with pytest.raises(UnboundLocalError):
        l1_out_double



def test_bridged_node_cleanup(tiny_model: NNsight):
    with tiny_model.session() as session:
        l = [].save()
        for item in [0, 1, 2]:
            if item == 2:
                break
            l.append(item)

    assert l == [0, 1]


def test_nested_iterator(tiny_model: NNsight):

    with tiny_model.session() as session:
        l = [].save()
        l.append([0])
        l.append([1])
        l.append([2])
        l2 = [].save()
        for item in l:
            for item_2 in item:
                l2.append(item_2)

    assert l2 == [0, 1, 2]


def test_nnsight_builtins(tiny_model: NNsight):
    with tiny_model.session() as session:
        nn_list = [].save()
        sesh_list = [].save()
        apply_list = [].save()

        for l in [nn_list, sesh_list, apply_list]:
            l.append(int)
            l.append("Hello World")
            l.append({"a": "1"})

    assert nn_list == sesh_list
    assert sesh_list == apply_list


def test_torch_creation_operations_patch(tiny_model: NNsight, tiny_input: torch.Tensor):
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

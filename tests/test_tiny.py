from collections import OrderedDict
from enum import auto

import pytest
import torch

import nnsight
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

@pytest.fixture(autouse=True)
def model_clear(tiny_model: NNsight):
    # clear the model before each test
    tiny_model._clear()
    return tiny_model

@pytest.fixture
def tiny_input():
    return torch.rand((1, input_size))


@torch.no_grad()
def test_tiny(tiny_model: NNsight, tiny_input: torch.Tensor):

    with tiny_model.trace(tiny_input):

        hs = tiny_model.layer2.output.save()

    assert isinstance(hs.value, torch.Tensor)


def test_grad_setting(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input, validate=True, scan=True):
        l1_grad = tiny_model.layer1.output.grad.clone().save()

        tiny_model.layer1.output.grad = (
            tiny_model.layer1.output.grad.clone() * 2
        )

        l1_grad_double = tiny_model.layer1.output.grad.save()

        loss = tiny_model.output.sum()
        loss.backward()

    assert torch.equal(l1_grad.value * 2, l1_grad_double.value)


def test_external_proxy_intervention_executed_locally(
    tiny_model: NNsight, tiny_input: torch.Tensor
):
    with tiny_model.session(validate=True) as sesh:
        with tiny_model.trace(tiny_input, validate=True, scan=True) as tracer_1:
            l1_out = tiny_model.layer1.output.save()

        with tiny_model.trace(tiny_input, validate=True, scan=True) as tracer_2:
            l1_out[:, 2] = 5

        assert list(tracer_2.graph.nodes.keys()) == [
            "BridgeProtocol_0",
            "setitem_0",
        ]

    assert l1_out[:, 2] == 5


def test_early_stop_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input, validate=True, scan=True):
        l1_out = tiny_model.layer1.output.save()
        l2_out = tiny_model.layer2.output.save()
        tiny_model.layer1.output.stop()

    assert isinstance(l1_out.value, torch.Tensor)

    with pytest.raises(ValueError):
        l2_out.value


def test_true_conditional_protocol(
    tiny_model: NNsight, tiny_input: torch.Tensor
):
    with tiny_model.trace(tiny_input, validate=True, scan=True) as tracer:
        num = 5
        with tracer.cond(num > 0):
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out.value, torch.Tensor)
    assert torch.all(l1_out.value == 1).item()


def test_false_conditional_protocol(
    tiny_model: NNsight, tiny_input: torch.Tensor
):
    with tiny_model.trace(tiny_input, validate=True, scan=True) as tracer:
        num = 5
        with tracer.cond(num < 0):
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

        # check that the condition does not persist outside the context
        l2_out = tiny_model.layer2.output.save()

    with pytest.raises(ValueError):
        l1_out.value
    assert isinstance(l2_out.value, torch.Tensor)


def test_node_as_condition(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test a Tensor a boolean value as a result of a boolean operation on an InterventionProxy"""

    with tiny_model.trace(tiny_input, validate=True, scan=True) as tracer:
        out = tiny_model.layer1.output
        out[:, 0] = 1
        with tracer.cond(out[:, 0] != 1):
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

    with pytest.raises(ValueError):
        l1_out.value


def test_multiple_dependent_conditionals(
    tiny_model: NNsight, tiny_input: torch.Tensor
):
    """Test that interventions defined within different Intervention contexts can be referenced if their conditions evaluated to True."""

    with tiny_model.trace(tiny_input, validate=True, scan=True) as tracer:
        num = 5
        l1_out = tiny_model.layer1.output
        l2_out = tiny_model.layer2.output.save()
        with tracer.cond(num > 0):
            l1_out[:] = 1

        with tracer.cond(l1_out[:, 0] != 1):
            tiny_model.layer2.output[:] = 2

        with tracer.cond(l1_out[:, 0] == 1):
            tiny_model.layer2.output[:] = 3

    assert torch.all(l2_out.value == 3).item()


def test_nested_conditionals(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input, validate=True, scan=True) as tracer:
        num = 5
        with tracer.cond(num > 0):  # True
            l1_out = tiny_model.layer1.output.save()

            with tracer.cond(num > 0):  # True
                tiny_model.layer1.output[:] = 1

            with tracer.cond(num < 0):  # False
                tiny_model.layer1.output[:] = 2

        with tracer.cond(num < 0):  # False
            tiny_model.layer2.output[:] = 0

            with tracer.cond(num > 0):  # True
                l2_out = tiny_model.layer2.output.save()

    assert isinstance(l1_out.value, torch.Tensor)
    assert torch.all(l1_out.value == 1).item()
    with pytest.raises(ValueError):
        l2_out.value


def test_conditional_trace(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session(validate=True) as session:
        num = 5
        with session.cond(num > 0):
            with tiny_model.trace(tiny_input, validate=True, scan=True):
                output = tiny_model.output.save()

    assert isinstance(output.value, torch.Tensor)


def test_conditional_iteration(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session(validate=True) as session:
        result = session.apply(list).save()
        with session.iter([0, 1, 2], return_context=True, validate=True) as (
            item,
            iterator,
        ):
            with iterator.cond(item % 2 == 0):
                with tiny_model.trace(tiny_input, validate=True, scan=True):
                    result.append(item)

    assert result.value == [0, 2]


def test_bridge_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session(validate=True) as session:
        val = session.apply(int, 0)
        with tiny_model.trace(tiny_input, validate=True, scan=True):
            tiny_model.layer1.output[:] = (
                val  # fetches the val proxy using the bridge protocol
            )
            l1_out = tiny_model.layer1.output.save()

    assert torch.all(l1_out.value == 0).item()


def test_update_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session(validate=True) as session:
        sum = session.apply(int, 0).save()
        with session.iter([0, 1, 2], validate=True) as item:
            sum.update(sum + item)

        sum.update(sum + 4)

        with tiny_model.trace(tiny_input, validate=True, scan=True):
            sum.update(sum + 3)
            double_sum = (sum * 2).save()

    assert double_sum.value == 20


def test_sequential_graph_based_context_exit(tiny_model: NNsight):
    with tiny_model.session(validate=True) as session:
        l = session.apply(list).save()
        l.append(0)

        with session.iter([1, 2, 3, 4], return_context=True, validate=True) as (
            item,
            iterator,
        ):
            with iterator.cond(item == 3):
                iterator.exit()
            l.append(item)
        l.append(5)
        session.exit()
        l.append(6)

    assert l.value == [0, 1, 2, 5]


def test_tracer_stop(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input, validate=True, scan=True):
        l1_out = tiny_model.layer1.output
        tiny_model.layer1.output.stop()
        l1_out_double = l1_out * 2

    with pytest.raises(ValueError):
        l1_out.value


def test_bridged_node_cleanup(tiny_model: NNsight):
    with tiny_model.session(validate=True) as session:
        l = session.apply(list)
        with session.iter([0, 1, 2], return_context=True, validate=True) as (item, iterator):
            with iterator.cond(item == 2):
                iterator.exit()
            l.append(item)

    with pytest.raises(ValueError):
        l.value


def test_nested_iterator(tiny_model: NNsight):

    with tiny_model.session(validate=True) as session:
        l = session.apply(list)
        l.append([0])
        l.append([1])
        l.append([2])
        l2 = session.apply(list).save()
        with session.iter(l, validate=True) as item:
            with session.iter(item, validate=True) as item_2:
                l2.append(item_2)

    assert l2.value == [0, 1, 2]

def test_nnsight_builtins(tiny_model: NNsight):
    with tiny_model.session() as session:
        nn_list = nnsight.list().save()
        sesh_list = session.list().save()
        apply_list = session.apply(list).save()

        with session.iter([nn_list, sesh_list, apply_list], return_context=True) as (l, iterator):
            l.append(nnsight.int(0))
            l.append(iterator.str("Hello World"))
            l.append(nnsight.dict({"a": "1"}))

    assert nn_list == sesh_list
    assert sesh_list == apply_list

def test_torch_creation_operations_patch(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input, scan=False, validate=False):
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

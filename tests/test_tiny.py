from collections import OrderedDict

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


@pytest.fixture
def tiny_input():
    return torch.rand((1, input_size))


@torch.no_grad()
def test_tiny(tiny_model: NNsight, tiny_input: torch.Tensor):

    with tiny_model.trace(tiny_input):

        hs = tiny_model.layer2.output.save()

    assert isinstance(hs.value, torch.Tensor)


def test_grad_setting(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input):
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
    with tiny_model.session() as sesh:
        with tiny_model.trace(tiny_input) as tracer_1:
            l1_out = tiny_model.layer1.output.save()

        with tiny_model.trace(tiny_input) as tracer_2:
            l1_out[:, 2] = 5

        assert list(tracer_2.graph.nodes.keys()) == [
            "BridgeProtocol_0",
            "setitem_0",
        ]

    assert l1_out[:, 2] == 5


def test_early_stop_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input):
        l1_out = tiny_model.layer1.output.save()
        l2_out = tiny_model.layer2.output.save()
        tiny_model.layer1.output.stop()

    assert isinstance(l1_out.value, torch.Tensor)

    with pytest.raises(ValueError):
        l2_out.value


def test_true_conditional_protocol(
    tiny_model: NNsight, tiny_input: torch.Tensor
):
    with tiny_model.trace(tiny_input) as tracer:
        num = tracer.apply(int, 5)
        with num > 0:
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out.value, torch.Tensor)
    assert torch.all(l1_out.value == 1).item()


def test_false_conditional_protocol(
    tiny_model: NNsight, tiny_input: torch.Tensor
):
    with tiny_model.trace(tiny_input) as tracer:
        num = tracer.apply(int, 5)
        with num < 0:
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

        # check that the condition does not persist outside the context
        l2_out = tiny_model.layer2.output.save()

    with pytest.raises(ValueError):
        l1_out.value
    assert isinstance(l2_out.value, torch.Tensor)


def test_node_as_condition(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test a Tensor a boolean value as a result of a boolean operation on an InterventionProxy"""

    with tiny_model.trace(tiny_input):
        out = tiny_model.layer1.output
        out[:, 0] = 1
        with out[:, 0] != 1:
            tiny_model.layer1.output[:] = 1
            l1_out = tiny_model.layer1.output.save()

    with pytest.raises(ValueError):
        l1_out.value


def test_multiple_dependent_conditionals(
    tiny_model: NNsight, tiny_input: torch.Tensor
):
    """Test that interventions defined within different Intervention contexts can be referenced if their conditions evaluated to True."""

    with tiny_model.trace(tiny_input) as tracer:
        num = tracer.apply(int, 5)
        l1_out = tiny_model.layer1.output
        l2_out = tiny_model.layer2.output.save()
        with num > 0:
            l1_out[:] = 1

        with l1_out[:, 0] != 1:
            tiny_model.layer2.output[:] = 2

        with l1_out[:, 0] == 1:
            tiny_model.layer2.output[:] = 3

    assert torch.all(l2_out.value == 3).item()


def test_nested_conditionals(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input) as tracer:
        num = tracer.apply(int, 5)
        with num > 0:  # True
            l1_out = tiny_model.layer1.output.save()

            with num > 0:  # True
                tiny_model.layer1.output[:] = 1

            with num < 0:  # False
                tiny_model.layer1.output[:] = 2

        with num < 0:  # False
            tiny_model.layer2.output[:] = 0

            with num > 0:  # True
                l2_out = tiny_model.layer2.output.save()

    assert isinstance(l1_out.value, torch.Tensor)
    assert torch.all(l1_out.value == 1).item()
    with pytest.raises(ValueError):
        l2_out.value


def test_conditional_trace(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as session:
        num = session.apply(int, 5)
        with num > 0:
            with tiny_model.trace(tiny_input):
                output = tiny_model.output.save()

    assert isinstance(output.value, torch.Tensor)


def test_conditional_iteration(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as session:
        result = session.apply(list).save()
        with session.iter([0, 1, 2]) as (item, iterator):
            with item % 2 == 0:
                with tiny_model.trace(tiny_input):
                    result.append(item)

    assert result.value == [0, 2]


def test_bridge_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as session:
        val = session.apply(int, 0)
        with tiny_model.trace(tiny_input):
            tiny_model.layer1.output[:] = (
                val  # fetches the val proxy using the bridge protocol
            )
            l1_out = tiny_model.layer1.output.save()

    assert torch.all(l1_out.value == 0).item()


def test_update_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as session:
        sum = session.apply(int, 0).save()
        with session.iter([0, 1, 2]) as (item, iterator):
            sum.update(sum + item)

        sum.update(sum + 4)

        with tiny_model.trace(tiny_input):
            sum.update(sum + 3)
            double_sum = (sum * 2).save()

    assert double_sum.value == 20


def test_sequential_graph_based_context_exit(tiny_model: NNsight):
    with tiny_model.session() as session:
        l = session.apply(list).save()
        l.append(0)

        with session.iter([1, 2, 3, 4]) as (item, iterator):
            with item == 3:
                iterator.exit()
            l.append(item)
        l.append(5)
        session.exit()
        l.append(6)

    assert l.value == [0, 1, 2, 5]


def test_tracer_stop(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input):
        l1_out = tiny_model.layer1.output
        tiny_model.layer1.output.stop()
        l1_out_double = l1_out * 2

    with pytest.raises(ValueError):
        l1_out.value


def test_bridged_node_cleanup(tiny_model: NNsight):
    with tiny_model.session() as session:
        l = session.apply(list)
        with session.iter([0, 1, 2]) as (item, iterator):
            with item == 2:
                iterator.exit()
            l.append(item)

    with pytest.raises(ValueError):
        l.value


def test_nested_iterator(tiny_model: NNsight):

    with tiny_model.session() as session:
        l = session.apply(list)
        l.append([0])
        l.append([1])
        l.append([2])
        l2 = session.apply(list).save()
        with session.iter(l) as (item, iterator):
            with session.iter(item) as (item_2, iterator_2):
                l2.append(item_2)

    assert l2.value == [0, 1, 2]

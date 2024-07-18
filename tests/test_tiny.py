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

        tiny_model.layer1.output.grad = tiny_model.layer1.output.grad.clone() * 2

        l1_grad_double = tiny_model.layer1.output.grad.save()

        loss = tiny_model.output.sum()
        loss.backward()

    assert torch.equal(l1_grad.value * 2, l1_grad_double.value)


def test_external_proxy_intervention_executed_locally(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as sesh:
        with tiny_model.trace(tiny_input) as tracer_1:
            l1_out = tiny_model.layer1.output.save()

        with tiny_model.trace(tiny_input) as tracer_2:
            l1_out[:, 2] = 5

        assert list(tracer_2.graph.nodes.keys()) == ['BridgeProtocol_0', 'setitem_0']
    
    assert l1_out[:, 2] == 5

def test_early_stop_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.trace(tiny_input):
        l1_out = tiny_model.layer1.output.save()
        l2_out = tiny_model.layer2.output.save()
        tiny_model.layer1.output.stop()

    assert isinstance(l1_out.value, torch.Tensor)

    with pytest.raises(ValueError):
        l2_out.value

def test_bridge_protocol(tiny_model: NNsight, tiny_input: torch.Tensor):
    with tiny_model.session() as session:
        val = session.apply(int, 0)
        with tiny_model.trace(tiny_input):
            tiny_model.layer1.output[:] = val # fetches the val proxy using the bridge protocol
            l1_out = tiny_model.layer1.output.save()

    assert torch.all(l1_out.value == 0).item()
    
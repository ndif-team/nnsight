"""Shells for modules another stage owns: served values pass, a call or a
parameter raises naming the owner."""

import pytest
import torch

from nnsight import NNsight
from nnsight.modeling.vllm.pp import PPModuleMap
from nnsight.modeling.vllm.pp_envoys import (
    RemoteModuleError,
    RemoteShell,
    graft_children,
    install_shells,
)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.proj(x))


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(2)])
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


@pytest.fixture
def stage0():
    """Stage 0 of two: it holds block 0; block 1 and the norm live on stage 1."""
    module_map = PPModuleMap(2)
    module_map.set_derived_owners({"blocks.0": 0, "blocks.1": 1, "norm": 1})
    local, meta = Stack(), Stack()
    installed = install_shells(local, meta, module_map, 0)
    model = NNsight(local)
    graft_children(model, meta, 0)
    return model, installed


def test_remote_modules_become_shells_and_local_ones_stay(stage0):
    model, installed = stage0
    assert installed == ["model.blocks.1", "model.norm"]
    assert isinstance(model.blocks[1]._module, RemoteShell)
    assert isinstance(model.blocks[1].proj._module, RemoteShell)  # grafted from the meta copy
    assert not isinstance(model.blocks[0]._module, RemoteShell)


def test_a_call_raises_naming_the_owner(stage0):
    model, _ = stage0
    with pytest.raises(RemoteModuleError, match="stage 1"):
        model.norm(torch.zeros(1, 4))
    with pytest.raises(RemoteModuleError, match="stage 1"):
        model.blocks[1].proj(torch.zeros(1, 4))
    assert model.blocks[0](torch.zeros(1, 4)).shape == (1, 4)


def test_a_parameter_raises_by_attribute_and_by_param(stage0):
    model, _ = stage0
    with pytest.raises(RemoteModuleError, match="weight"):
        model.blocks[1].proj.weight
    with pytest.raises(RemoteModuleError, match="weight"):
        model.blocks[1].proj.param("weight")
    with pytest.raises(RemoteModuleError, match="bias"):
        model.norm.bias
    assert model.blocks[0].proj.param("weight") is model._module.blocks[0].proj.weight


class _FakeListener:
    def __init__(self):
        self.requests = []

    def begin_pull(self, owner, provider, req_id=None):
        self.requests.append((owner, provider, req_id))
        value = torch.full((4,), 3.0)

        class Pull:
            def complete(self, timeout=None):
                return value

        return Pull()


def test_param_pulls_from_the_owner_while_the_attribute_still_raises():
    module_map = PPModuleMap(2)
    module_map.set_derived_owners({"blocks.0": 0, "blocks.1": 1, "norm": 1})
    local, meta = Stack(), Stack()
    listener = _FakeListener()
    install_shells(local, meta, module_map, 0, listener)
    model = NNsight(local)
    graft_children(model, meta, 0, listener)
    assert torch.equal(model.norm.param("weight"), torch.full((4,), 3.0))
    assert torch.equal(model.blocks[1].proj.param("bias"), torch.full((4,), 3.0))
    assert listener.requests == [
        (1, "model.norm.param.weight", None),
        (1, "model.blocks.1.proj.param.bias", None),
    ]
    with pytest.raises(RemoteModuleError, match="weight"):
        model.norm.weight
    with pytest.raises(AttributeError, match="no parameter"):
        model.norm.param("nope")


def test_an_attribute_neither_side_has_is_a_plain_attribute_error(stage0):
    model, _ = stage0
    with pytest.raises(AttributeError):
        model.norm.no_such_attribute

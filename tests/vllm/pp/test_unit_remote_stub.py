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
from nnsight.modeling.vllm.pp_transport import Kept


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


def test_a_call_without_a_link_raises_naming_the_owner(stage0):
    model, _ = stage0
    with pytest.raises(RemoteModuleError, match="stage 1"):
        model.norm(torch.zeros(1, 4))
    with pytest.raises(RemoteModuleError, match="stage 1"):
        model.blocks[1].proj(torch.zeros(1, 4))
    assert model.blocks[0](torch.zeros(1, 4)).shape == (1, 4)


def test_a_call_runs_the_owner_state_here_without_touching_the_meta_copy():
    module_map = PPModuleMap(2)
    module_map.set_derived_owners({"blocks.0": 0, "blocks.1": 1, "norm": 1})
    owner = Stack()  # what stage 1 holds
    local, meta = Stack(), Stack()
    link = _FakeLink(state_of=owner.norm)
    install_shells(local, meta, module_map, 0, link)
    model = NNsight(local)
    graft_children(model, meta, 0, link)
    copy = model.norm._module._pp_meta
    copy.weight.output_dim = 0  # what vLLM stamps on a sharded parameter
    before = {name: (id(t), t.device, dict(vars(t))) for name, t in list(copy.named_parameters()) + list(copy.named_buffers())}
    x = torch.randn(2, 4)
    expected = owner.norm(x)
    got = model.norm(x)
    assert torch.allclose(got, expected)
    assert link.requests[-1] == (1, "model.norm.state")
    # The same parameter objects, on the same device, with the same attributes.
    after = {name: (id(t), t.device, dict(vars(t))) for name, t in list(copy.named_parameters()) + list(copy.named_buffers())}
    assert after == before


def test_param_after_a_call_still_carries_the_sharding_stamp():
    module_map = PPModuleMap(2)
    module_map.set_derived_owners({"blocks.0": 0, "blocks.1": 1, "norm": 1})
    owner = Stack()
    local, meta = Stack(), Stack()
    meta.norm.weight.output_dim = 0
    link = _FakeLink(state_of=owner.norm)
    install_shells(local, meta, module_map, 0, link)
    model = NNsight(local)
    graft_children(model, meta, 0, link)
    model.norm(torch.randn(2, 4))
    pulled = model.norm.param("weight")
    assert pulled.output_dim == 0


def test_a_fetch_outside_any_request_is_kept_for_the_engine():
    module_map = PPModuleMap(2)
    module_map.set_derived_owners({"blocks.0": 0, "blocks.1": 1, "norm": 1})
    local, meta = Stack(), Stack()
    link = _FakeLink()
    install_shells(local, meta, module_map, 0, link)
    model = NNsight(local)
    graft_children(model, meta, 0, link)
    model.norm.param("weight")
    model.norm.param("weight")
    assert len(link.requests) == 1  # answered from the kept copy the second time
    assert "model.norm.param.weight" in link.kept.pinned


class PackedHead(torch.nn.Module):
    """A head stored the way a quantized checkpoint stores one: packed
    integers and scales, and no ``weight``."""

    def __init__(self):
        super().__init__()
        self.register_buffer("qweight", torch.arange(8, dtype=torch.int32).reshape(2, 4))
        self.scales = torch.nn.Parameter(torch.full((4,), 0.5), requires_grad=False)


class PackedStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(4, 4)
        self.output_projection = PackedHead()


def test_a_head_kept_by_its_state_answers_param_under_its_own_names():
    """The head is fetched at load as its whole state, since its parameter names
    depend on the checkpoint; ``param`` is answered from it, and a name the
    head does not have raises instead of asking the owner."""
    module_map = PPModuleMap(2)
    module_map.set_derived_owners({"trunk": 0, "output_projection": 1})
    owner = PackedStack()
    owner.output_projection.qweight += 100
    local, meta = PackedStack(), PackedStack()
    link = _FakeLink(state_of=owner.output_projection)
    install_shells(local, meta, module_map, 0, link)
    model = NNsight(local)
    graft_children(model, meta, 0, link)

    model.output_projection._module._fetch("model.output_projection.state")  # as at load

    assert torch.equal(model.output_projection.param("qweight"), owner.output_projection.qweight)
    assert torch.equal(model.output_projection.param("scales"), owner.output_projection.scales)
    assert link.requests == [(1, "model.output_projection.state")]
    assert "model.output_projection.state" in link.kept.pinned
    with pytest.raises(AttributeError, match="weight"):
        model.output_projection.param("weight")


def test_a_parameter_raises_by_attribute_and_by_param(stage0):
    model, _ = stage0
    with pytest.raises(RemoteModuleError, match="weight"):
        model.blocks[1].proj.weight
    with pytest.raises(RemoteModuleError, match="weight"):
        model.blocks[1].proj.param("weight")
    with pytest.raises(RemoteModuleError, match="bias"):
        model.norm.bias
    assert model.blocks[0].proj.param("weight") is model._module.blocks[0].proj.weight


class _FakeLink:
    """Answers a parameter request with a constant and a state request with a
    given module's state."""

    def __init__(self, state_of=None):
        self.requests = []
        self.kept = Kept()
        self.state_of = state_of

    def request(self, owner, provider):
        self.requests.append((owner, provider))
        if provider.endswith(".state"):
            return dict(self.state_of.state_dict())
        return torch.full((4,), 3.0)


def test_param_pulls_from_the_owner_while_the_attribute_still_raises():
    module_map = PPModuleMap(2)
    module_map.set_derived_owners({"blocks.0": 0, "blocks.1": 1, "norm": 1})
    local, meta = Stack(), Stack()
    link = _FakeLink()
    install_shells(local, meta, module_map, 0, link)
    model = NNsight(local)
    graft_children(model, meta, 0, link)
    assert torch.equal(model.norm.param("weight"), torch.full((4,), 3.0))
    assert torch.equal(model.blocks[1].proj.param("bias"), torch.full((4,), 3.0))
    assert link.requests == [
        (1, "model.norm.param.weight"),
        (1, "model.blocks.1.proj.param.bias"),
    ]
    with pytest.raises(RemoteModuleError, match="weight"):
        model.norm.weight
    with pytest.raises(AttributeError, match="no parameter"):
        model.norm.param("nope")


def test_an_attribute_neither_side_has_is_a_plain_attribute_error(stage0):
    model, _ = stage0
    with pytest.raises(AttributeError):
        model.norm.no_such_attribute

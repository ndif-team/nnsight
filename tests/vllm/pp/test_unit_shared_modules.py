"""Persistent ids for a module the tree holds under several paths."""

import torch

from nnsight import NNsight
from nnsight.modeling.mixins.remotable import Remotable


class Block(torch.nn.Module):
    def __init__(self, act):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.act = act

    def forward(self, x):
        return self.act(self.proj(x))


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        act = torch.nn.GELU()  # one instance, every block
        self.blocks = torch.nn.ModuleList([Block(act) for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_every_alias_path_resolves_to_the_shared_module():
    model = NNsight(Stack())
    objects = Remotable._remoteable_persistent_objects(model)
    act = model._module.blocks[0].act
    for index in range(3):
        assert objects[f"Module:model.blocks.{index}.act"] is act

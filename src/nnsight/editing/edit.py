from typing import List

import torch

from ..envoy import Envoy

class Edit():

    def __init__(
        self,
        base: Envoy,
        name: str,
        stuff: torch.nn.Module
    ):
        self.base = base
        self.name = name
        self.stuff = stuff

    def edit(self):
        self.base._add_envoy(
            self.stuff,
            self.name
        )
        
        setattr(
            self.base,
            self.name,
            self.stuff
        )

def apply_edits(edits: List[Edit]):
    for edit in edits:
        edit.edit()
from __future__ import annotations
from abc import abstractmethod
from typing import List

import torch

class Edit:

    @abstractmethod
    def edit(self, model: torch.nn.Module):
        pass

    @abstractmethod
    def restore(self, model: torch.nn.Module):
        pass


class Editor:
    def __init__(self, model: torch.nn.Module, edits: List[Edit]) -> None:
        self.model = model
        self.edits = edits

    def __enter__(self) -> Editor:
        for edit in self.edits:
            edit.edit(self.model)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for edit in self.edits:
            edit.restore(self.model)

from __future__ import annotations
from abc import abstractmethod
from typing import List

import torch

class Edit:

    @abstractmethod
    def edit(self, obj: torch.nn.Module):
        pass

    @abstractmethod
    def restore(self, obj: torch.nn.Module):
        pass


class Editor:
    def __init__(self, obj: object, edits: List[Edit]) -> None:
        self.obj = obj
        self.edits = edits

    def __enter__(self) -> Editor:
        for edit in self.edits:
            edit.edit(self.obj)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for edit in self.edits:
            edit.restore(self.obj)

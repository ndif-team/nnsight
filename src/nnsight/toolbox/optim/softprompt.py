from typing import Any
import torch

from ...module import Module
from . import Optimization


class SoftPrompt(Optimization):
    def __init__(self, module: Module, n: int) -> None:
        self.module = module
        self.n = n

        self.embedding = torch.nn.Parameter(
            torch.zeros((self.n, self.module.embedding_dim)), requires_grad=True
        )

    def __call__(self) -> Any:
        self.module.output = self.embedding[:]

    def parameters(self):
        return [self.embedding]

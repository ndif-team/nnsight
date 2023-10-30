from typing import Any
import torch

from ...module import Module
from . import Optimization


class LORA(Optimization):
    def __init__(self, module: Module, r: int) -> None:
        self.module = module
        self.r = r

        self.WA = torch.nn.Parameter(torch.empty(self.module.input_shape[0][-1], self.r), requires_grad=True)
        self.WB = torch.nn.Parameter(torch.empty(self.r, self.module.output_shape[-1]), requires_grad=True)

    def __call__(self, alpha:float=1.0) -> Any:

        inp = self.module.input[0]

        self.module.output = (torch.matmul(torch.matmul(inp, self.WA), self.WB) + self.module.output) * alpha

    def parameters(self):
        return [self.WA, self.WB]

from typing import Iterator

import torch
from torch.nn.parameter import Parameter
from typing_extensions import Self

from .tracing.Proxy import Proxy


class Module(torch.nn.Module):

    def save(self) -> Self:

        [param.save() for param in self.parameters()]

        return self

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return [
            param
            for param in self.__dict__.values()
            if isinstance(param, Proxy)
            and param.node.target is torch.nn.parameter.Parameter
        ]

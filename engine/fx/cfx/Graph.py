import inspect
from typing import Any, Dict, List

import torch

from .Node import Node
from .Patcher import Patcher
from .Proxy import Proxy


class Graph:
    @staticmethod
    def trace(module: torch.nn.Module, *args: List[Any]):
        graph = Graph()

        forward = module.__class__.forward

        args = list(args)

        signature = inspect.signature(forward)
        args = args + [
            param.default
            for param in list(signature.parameters.values())[len(args) + 1 :]
        ]
        # kwargs ?
        args = [
            graph.proxy(graph=graph, value=arg, target='parameter')
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in args
        ]

        module_proxy = graph.proxy(graph=graph, value=module, target='module', args=args)

        with Patcher() as patcher:
            patcher.patch(torch.full)
            patcher.patch(torch.finfo)
            patcher.patch(torch.arange)
            forward(module_proxy, *args)

        return graph

    def __init__(self) -> None:
        self.nodes: List[Node] = list()

    def proxy(self, *args, **kwargs) -> Proxy:
        node = Node(*args, **kwargs)

        print(node)

        self.nodes.append(node)

        return Proxy(node)

    def is_module_node(self, value):
        return isinstance(value, Node) and isinstance(value.value, torch.nn.Module)

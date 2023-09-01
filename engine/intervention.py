from __future__ import annotations

from typing import Any, Union

import torch.futures

from . import util
from .fx.Graph import Graph
from .fx.Node import Node
from .fx.Proxy import Proxy


class InterventionProxy(torch.futures.Future, Proxy):
    @staticmethod
    def proxy_save(value: Any) -> None:
        return util.apply(value, lambda x: x.clone(), torch.Tensor)

    def __init__(self, *args, **kwargs):
        Proxy.__init__(self, *args, **kwargs)
        torch.futures.Future.__init__(self)

    def set(self, value: Union[InterventionProxy, Any]):
        self.node.graph.add(
            graph=self.node.graph,
            value=Proxy.get_value(value),
            target=Node.update,
            args=[self.node, value],
        )

    def save(self) -> InterventionProxy:
        proxy = self.node.graph.add(
            graph=self.node.graph,
            value=self.node.proxy_value,
            target=InterventionProxy.proxy_save,
            args=[self.node],
        )

        self.node.graph.add(
            graph=self.node.graph,
            value=None,
            target="null",
            args=[proxy.node],
        )

        return proxy

    def token(self, idx: int) -> InterventionProxy:
        if idx >= 0:
            n_tokens = self.node.proxy_value.shape[1]
            idx = -(n_tokens - idx)

        return self[:, idx]

    def t(self, idx: int) -> InterventionProxy:
        return self.token(idx)

    @property
    def shape(self):
        return util.apply(self.node.proxy_value, lambda x : x.shape, torch.Tensor)

    @property
    def value(self):
        return self.node.future.value()


def intervene(activations, module_path: str, graph: Graph, key: str):
    batch_idx = 0

    module_path = f"{module_path}.{key}.{graph.generation_idx}"

    batch_module_path = f"{module_path}.{batch_idx}"

    while batch_module_path in graph.argument_node_names:
        node: Node = graph.nodes[graph.argument_node_names[batch_module_path]]

        node.future.set_result(
            util.apply(activations, lambda x: x[[batch_idx]], torch.Tensor)
        )

        batch_idx += 1

        batch_module_path = f"{module_path}.{batch_idx}"

    return activations

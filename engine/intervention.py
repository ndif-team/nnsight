from __future__ import annotations

from typing import Any, Union

import torch.futures

from . import util
from .fx.Graph import Graph
from .fx.Node import Node
from .fx.Proxy import Proxy


class InterventionProxy(Proxy):
    @staticmethod
    def proxy_save(value: Any) -> None:
        return util.apply(value, lambda x: x.clone(), torch.Tensor)

    def set(self, value: Union[InterventionProxy, Any]):

        Node.update(self.node.proxy_value, Proxy.get_value(value))

        self.node.graph.add(
            graph=self.node.graph,
            value=self.node.proxy_value,
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
        return util.apply(self.node.proxy_value, lambda x: x.shape, torch.Tensor)

    @property
    def value(self):
        return self.node.future.value()


def intervene(activations, module_path: str, graph: Graph, key: str):
    """Entry to intervention graph. This should be hooked to all modules involved in intervention graph.

    Args:
        activations (_type_): _description_
        module_path (str): _description_
        graph (Graph): _description_
        key (str): _description_

    Returns:
        _type_: _description_
    """
    batch_idx = 0

    # Key to module activation argument nodes has format: <module path>.<output/input>.<generation index>.<batch index>
    module_path = f"{module_path}.{key}.{graph.generation_idx}"

    batch_module_path = f"{module_path}.{batch_idx}"

    # We create a new key as we increment batch_idx and check if that key is in the graph's argument_node_names dict.
    while batch_module_path in graph.argument_node_names:
        # If it exists, we grab it.
        node = graph.nodes[graph.argument_node_names[batch_module_path]]

        # We set its result to the activatins, indexed by only the relevant batch index.
        node.future.set_result(
            util.apply(activations, lambda x: x.select(0, batch_idx).unsqueeze(0), torch.Tensor)
        )

        # Increment batch_idx and go again.
        batch_idx += 1

        batch_module_path = f"{module_path}.{batch_idx}"

    return activations

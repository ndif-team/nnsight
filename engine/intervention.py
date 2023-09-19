from __future__ import annotations

from typing import Any, Union

import torch.futures

from . import util
from .fx.Graph import Graph
from .fx.Node import Node
from .fx.Proxy import Proxy


class TokenIndexer:
    def __init__(self, proxy: InterventionProxy) -> None:
        self.proxy = proxy

    def convert_idx(self, idx: int):
        if idx >= 0:
            n_tokens = self.proxy.node.proxy_value.shape[1]
            idx = -(n_tokens - idx)

        return idx

    def __getitem__(self, key: int) -> Proxy:

        key = self.convert_idx(key)

        return self.proxy[:, key]

    def __setitem__(self, key: int, value: Union[Proxy, Any]) -> None:
        key = self.convert_idx(key)

        self.proxy[:, key] = value


class InterventionProxy(Proxy):
    @staticmethod
    def proxy_save(value: Any) -> None:
        return util.apply(value, lambda x: x.clone(), torch.Tensor)

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

    @property
    def token(self) -> TokenIndexer:
        return TokenIndexer(self)

    @property
    def t(self) -> TokenIndexer:
        return self.token

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

    # Key to module activation argument nodes has format: <module path>.<output/input>.<generation index>.<batch index>
    module_path = f"{module_path}.{key}.{graph.generation_idx}"

    if module_path in graph.argument_node_names:

        argument_node_names = graph.argument_node_names[module_path]

        # multiple argument nodes can have same module_path if there are multiple invocations.
        for argument_node_name in argument_node_names:

            node = graph.nodes[argument_node_name]

            # args for argument nodes are (module_path, batch_size, batch_start)
            _, batch_size, batch_start = node.args

            # We set its result to the activations, indexed by only the relevant batch idxs.
            node.future.set_result(
                util.apply(
                    activations, lambda x: x.narrow(0, batch_start, batch_size), torch.Tensor
                )
            )

    return activations

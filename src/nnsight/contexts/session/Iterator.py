from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Tuple

from ... import util
from ...tracing import protocols
from ...tracing.Graph import Graph
from .Collection import Collection

if TYPE_CHECKING:
    from ...intervention import InterventionProxy
    from ...tracing.Node import Node
    from ...tracing.Proxy import Proxy


class IteratorItemProtocol(protocols.Protocol):

    @classmethod
    def add(cls, graph: Graph, value: Any) -> "Proxy":

        return graph.create(target=cls, proxy_value=value, args=[None])

    @classmethod
    def set(cls, graph: Graph, value: Any) -> None:

        graph.nodes[f"{cls.__name__}_0"].args[0] = value

    @classmethod
    def execute(cls, node: protocols.Node):

        value = node.prepare_inputs(node.args[0])

        node.set_value(value)


class StatDefaultProtocol(protocols.Protocol):

    @classmethod
    def add(cls, graph: Graph, default_value: Any):

        return graph.create(target=cls, proxy_value=default_value, args=[default_value])

    @classmethod
    def execute(cls, node: protocols.Node):

        node.set_value(node.args[0])


class StatUpdateProtocol(protocols.Protocol):

    @classmethod
    def add(cls, graph: Graph, default_value_node: "Node", update_value: Any):

        return graph.create(
            target=cls,
            proxy_value=default_value_node.proxy_value,
            args=[
                default_value_node.graph.id,
                default_value_node.name,
                update_value,
            ],
        )

    @classmethod
    def execute(cls, node: "Node"):

        bridge = protocols.BridgeProtocol.get_bridge(node.graph)

        default_node_graph_id, default_node_name, update_value = node.args

        default_node: "Node" = bridge.id_to_graph[default_node_graph_id].nodes[
            default_node_name
        ]

        default_node._value = util.apply(
            update_value, lambda x: x.value, type(default_node)
        )

        if bridge.release:

            node.set_value(default_node.value)


class Stat:

    def __init__(self, graph: Graph) -> None:

        self.graph = graph
        self.proxy: Proxy = None

    def default(self, value: Any) -> InterventionProxy:

        # TODO error if already called.

        self.proxy = StatDefaultProtocol.add(self.graph, value)

        return self.proxy

    def update(self, value: Proxy) -> InterventionProxy:

        return StatUpdateProtocol.add(self.graph, self.proxy.node, value)


class Iterator(Collection):

    def __init__(self, data: Iterable, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.data = data

    def __enter__(self) -> Tuple[int, Iterator]:

        super().__enter__()

        iter_item_proxy = IteratorItemProtocol.add(self.graph, next(iter(self.data)))

        return iter_item_proxy, self

    def stat(self) -> Stat:

        return Stat(self.accumulator.graph)

    ### BACKENDS ########

    def local_backend_execute(self) -> None:

        bridge = protocols.BridgeProtocol.get_bridge(self.graph)

        bridge.locks += 1

        last_idx = len(self.data) - 1

        for idx, item in enumerate(self.data):

            last_iter = idx == last_idx

            if last_iter:

                bridge.locks -= 1

            IteratorItemProtocol.set(self.graph, item)

            super().local_backend_execute()

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Tuple

from ...tracing import protocols
from ...tracing.Graph import Graph
from .Collection import Collection

if TYPE_CHECKING:
    from ...intervention import InterventionProxy
    from ...tracing.Node import Node
    from ...tracing.Proxy import Proxy


class SumProtocol(protocols.Protocol):

    name = "iter_sum"

    @classmethod
    def add(cls, graph: Graph, value_node: "Node") -> "InterventionProxy":

        # Check  if value node's Graph is exited? Otherwise that should be an error

        return graph.create(
            target=cls.name, proxy_value=value_node.proxy_value, args=[value_node]
        )

    @classmethod
    def execute(cls, node: "Node"):

        value_node: "Node" = node.args[0]

        if not node.done():

            node._value = value_node.value

        else:

            node._value += value_node.value

        node.reset()

        if protocols.BridgeProtocol.get_bridge(node.graph).release:

            node.set_value(node._value)


class IteratorItemProtocol(protocols.Protocol):

    attachment_name = "nnsight_iter_idx"

    @classmethod
    def add(cls, graph: Graph, value: Any) -> "Proxy":

        return graph.create(target=cls, proxy_value=value)

    @classmethod
    def idx(cls, graph: Graph) -> int:

        if not cls.attachment_name in graph.attachments:

            graph.attachments[cls.attachment_name] = 0

        else:

            graph.attachments[cls.attachment_name] += 1

        return graph.attachments[cls.attachment_name]

    @classmethod
    def set(cls, graph: Graph, value: Any, iter_idx: int) -> None:

        graph.nodes[f"{cls.__name__}_{iter_idx}"].set_value(value)


class Iterator(Collection):

    def __init__(self, data: Iterable, *args, iter_idx: int = None, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.data = data
        self.iter_idx = IteratorItemProtocol.idx(self.accumulator.graph)

    def __enter__(self) -> Tuple[int, Iterator]:

        super().__enter__()

        iter_item_proxy = IteratorItemProtocol.add(
            self.accumulator.graph, next(iter(self.data))
        )

        return iter_item_proxy, self

    ### BACKENDS ########

    def local_backend_execute(self) -> None:

        self.accumulator.graph.compile()

        self.accumulator.bridge.release = False

        last_idx = len(self.data) - 1

        for idx, item in enumerate(self.data):

            last_iter = idx == last_idx

            if last_iter:

                self.accumulator.bridge.release = True

            IteratorItemProtocol.set(self.accumulator.graph, item, self.iter_idx)

            self.iterator_backend_execute(last_iter=last_iter)

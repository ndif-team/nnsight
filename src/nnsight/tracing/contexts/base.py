from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Generic, Optional, Type

from typing_extensions import Self

from ...tracing.graph import Node, NodeType, Proxy, ProxyType
from ..backends import Backend, ExecutionBackend
from ..graph import Graph, GraphType, SubGraph
from ..protocols import Protocol


class Context(Protocol, AbstractContextManager, Generic[GraphType]):

    def __init__(
        self,
        *args,
        backend: Optional[Backend] = None,
        parent: Optional[GraphType] = None,
        graph: Optional[GraphType] = None,
        graph_class: Type[SubGraph] = SubGraph,
        node_class: Type[NodeType] = Node,
        proxy_class: Type[ProxyType] = Proxy,
        **kwargs,
    ) -> None:

        if backend is None and parent is None:
            backend = ExecutionBackend()

        self.backend = backend

        if parent is None:
            parent = Graph(node_class=node_class, proxy_class=proxy_class)
            parent.stack.append(parent)

        self.graph = graph_class(*args, parent, **kwargs)

        self.graph.stack.append(self.graph)

        if graph is not None:
            graph.copy(self.graph)

        self.args = []
        self.kwargs = {}

    def __enter__(self) -> Self:

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        graph = self.graph.stack.pop()

        if isinstance(exc_val, BaseException):
            raise exc_val

        self.add(graph.stack[-1], graph, *self.args, **self.kwargs)

        if self.backend is not None:
            
            graph = graph.stack.pop()
            
            graph.alive = False

            self.backend(graph)

    @classmethod
    def execute(cls, node: NodeType):

        graph: GraphType = node.args[0]

        graph.reset()
        graph.execute()

        node.set_value(None)
    
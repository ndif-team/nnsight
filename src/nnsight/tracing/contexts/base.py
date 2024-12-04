from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Generic, Optional, Type

from typing_extensions import Self

from ... import CONFIG
from ...tracing.graph import Node, NodeType, Proxy, ProxyType
from ..backends import Backend, ExecutionBackend
from ..graph import Graph, GraphType, SubGraph, viz_graph
from ..protocols import Protocol

class Context(Protocol, AbstractContextManager, Generic[GraphType]):
    """A `Context` represents a scope (or slice) of a computation graph with specific logic for adding and executing nodes defined within it.
    It has a `SubGraph` which contains the nodes that make up the operations of the context.
    As an `AbstractContextManager`, entering adds its sub-graph to the stack, making new nodes created while within this context added to it's sub-graph.
        Exiting pops its sub-graph off the stack, allowing nodes to be added to its parent, and adds itself as a node to its parent `Context`/`SubGraph`. ( To say, "execute me")
        If the `Context` has a backend, it pops its parent off the stack and passes it to the `Backend` object to execute.
        (This only happens if the context is the root-most context, and its parent is therefore the root `Graph`)
    As a `Context` is itself a `Protocol`, it defines how to execute it's sub-graph in the `execute` method.
    

    Attributes:
    
        backend (Backend): Backend to execute the deferred root computation graph
    """

    def __init__(
        self,
        *args,
        backend: Optional[Backend] = None,
        parent: Optional[GraphType] = None,
        graph: Optional[GraphType] = None,
        graph_class: Type[SubGraph] = SubGraph,
        node_class: Type[NodeType] = Node,
        proxy_class: Type[ProxyType] = Proxy,
        debug: bool = False,
        **kwargs,
    ) -> None:

        # If this is the root graph, we want to execute it upon exit.
        # Otherwise its a child context/graph and all we want to
        if backend is None and parent is None:
            backend = ExecutionBackend(injection=CONFIG.APP.FRAME_INJECTION)

        self.backend = backend

        if parent is None:
            parent = Graph(node_class=node_class, proxy_class=proxy_class, debug=debug)
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

    def vis(self, *args, **kwargs):
        viz_graph(self.graph, *args, **kwargs)

    @classmethod
    def execute(cls, node: NodeType):

        graph: GraphType = node.args[0]

        graph.reset()
        graph.execute()

        node.set_value(None)
    
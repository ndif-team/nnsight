from __future__ import annotations

import weakref
from contextlib import AbstractContextManager
from typing import Any, Callable, Optional, Union

from typing_extensions import Self

from ...tracing.graph.node import Node

from ..backends import Backend
from ..graph import Graph, SubGraph
from ..protocols import Protocol


class Context(Protocol, AbstractContextManager):

    def __init__(
        self,
        backend: Optional[Backend] = None,
        parent: Optional[Graph] = None,
        graph: Optional[Graph] = None,
        **kwargs,
    ) -> None:

        self.backend = Backend() if backend is None else backend

        if parent is None:
            parent = Graph()
            parent.stack.append(parent)

        self.graph: Graph = SubGraph(parent, **kwargs) if graph is None else graph

        self.args = []
        self.kwargs = {}

    def __enter__(self) -> Self:

        self.graph.stack.append(self.graph)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.graph.stack.pop()

        if isinstance(exc_val, BaseException):
            self.graph = None
            raise exc_val

        self.backend(self)

    @classmethod
    def execute(cls, node: Node):

        graph: Graph = node.args[0]

        graph.reset()
        graph.execute()
        
        node.set_value(None)

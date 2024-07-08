from __future__ import annotations

import weakref
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from typing_extensions import Self

from ...tracing import protocols
from ...tracing.Bridge import Bridge
from ...tracing.Graph import Graph
from ..backends import Backend, BridgeMixin, LocalMixin


class Collection(AbstractContextManager, LocalMixin, BridgeMixin):
    """A Collection is a collection of objects to execute.

    Attributes:

        graph (Graph): __desc__
        backend (Backend): Backend to execute this Collection on __exit__.
    """

    def __init__(
        self,
        backend: Backend,
        bridge: Bridge,
        graph: Graph = None,
        validate: bool = False,
    ) -> None:

        self.graph = Graph(validate=validate) if graph is None else graph

        if bridge is not None:

            bridge.add(self.graph)

        self.backend = backend
        
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(exc_val, BaseException):
            raise exc_val

        self.backend(self)

    ### BACKENDS ########

    def local_backend_execute(self) -> None:

        self.graph.compile()

        graph = self.graph
        graph.alive = False

        if not isinstance(graph, weakref.ProxyType):
            self.graph = weakref.proxy(graph)

        return graph

    def bridge_backend_handle(self, bridge: Bridge) -> None:

        bridge.pop_graph()

        protocols.LocalBackendExecuteProtocol.add(self, bridge.peek_graph())

        self.graph = weakref.proxy(self.graph)

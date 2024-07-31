from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

from ...tracing.Bridge import Bridge
from ...tracing.Graph import Graph
from ..backends import Backend, BridgeBackend, RemoteMixin
from ..GraphBasedContext import GraphBasedContext
from .Iterator import Iterator

if TYPE_CHECKING:
    from ...models.mixins import RemoteableMixin
    from ...models.NNsightModel import NNsight


class Session(GraphBasedContext, RemoteMixin):
    """A Session is a root Collection that handles adding new Graphs and new Collections while in the session.

    Attributes:
        bridge (Bridge): Bridge object which stores all Graphs added during the session and handles interaction between them
        graph (Graph): Root Graph where operations and values meant for access by all subsequent Graphs should be stored and referenced.
        model (NNsight): NNsight model.
        backend (Backend): Backend for this context object.
    """

    def __init__(
        self, backend: Backend, model: "NNsight", *args, bridge: Bridge = None, **kwargs
    ) -> None:

        self.bridge = Bridge() if bridge is None else bridge

        self.model = model

        GraphBasedContext.__init__(self, backend, bridge=self.bridge, proxy_class=model.proxy_class, *args, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.model._session = None

        if isinstance(exc_val, BaseException):
            raise exc_val

        self.backend(self)

    def iter(self, iterable) -> Iterator:

        bridge = weakref.proxy(self.bridge)

        backend = BridgeBackend(bridge)

        return Iterator(iterable, backend, bridge=bridge, proxy_class=self.model.proxy_class)

    ### BACKENDS ########

    def local_backend_execute(self) -> Dict[int, Graph]:

        super().local_backend_execute()

        local_result = self.bridge.id_to_graph

        self.bridge = weakref.proxy(self.bridge)

        return local_result

    def remote_backend_get_model_key(self) -> str:

        self.model: "RemoteableMixin"

        return self.model.to_model_key()

    def remote_backend_postprocess_result(self, local_result: Dict[int, Graph]):

        from ...schema.Response import ResultModel

        return {id: ResultModel.from_graph(graph) for id, graph in local_result.items()}

    def remote_backend_handle_result_value(self, value: Dict[int, Dict[str, Any]]):

        for graph_id, saves in value.items():

            graph = self.bridge.id_to_graph[graph_id]

            for node_name, node_value in saves.items():
                graph.nodes[node_name]._value = node_value

            graph.alive = False

        self.bridge = weakref.proxy(self.bridge)

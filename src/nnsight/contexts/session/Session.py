from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

from ...tracing import protocols
from ...tracing.Bridge import Bridge
from ...tracing.Graph import Graph
from ..backends import SessionBackend, Backend
from .Collection import Collection
from .Iterator import Iterator

if TYPE_CHECKING:
    from ...models.mixins import RemoteableMixin
    from ...models.NNsightModel import NNsight


class Session(Collection):
    """A Session is a root Collection that handles adding new Graphs and new Collections while in the session.

    Attributes:
        bridge (Bridge): Bridge object which stores all Graphs added during the session and handles interaction between them
        graph (Graph): Root Graph where operations and values meant for access by all subsequent Graphs should be stored and referenced.
        model (NNsight): NNsight model.
        backend (Backend): Backend for this context object.
        collector_stack (List[Collection]): Stack of all Collections added during the session to keep track of which Collection to add a Tracer to when calling model.trace().
    """

    def __init__(self, backend: Backend, model: "NNsight", graph: Graph = None) -> None:

        self.bridge = Bridge()
        self.graph = (
            Graph(proxy_class=model.proxy_class, validate=False)
            if graph is None
            else graph
        )

        protocols.BridgeProtocol.set_bridge(self.graph, self.bridge)

        self.bridge.add(self.graph)

        self.collector_stack: List[Collection] = list()

        self.model = model

        Collection.__init__(self, SessionBackend(self), self)

        self.backend = backend

    def __enter__(self) -> Session:

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.model._session = None

        if isinstance(exc_val, BaseException):
            raise exc_val

        self.backend(self)

    def iter(self, iterable) -> Iterator:

        backend = SessionBackend(self)

        return Iterator(iterable, backend, self)

    ### BACKENDS ########
    
    def local_backend_execute(self) -> Dict[int, Graph]:
        
        super().local_backend_execute()
        
        local_result = self.bridge.id_to_graph
        
        self.bridge = None
        
        return local_result

    def remote_backend_get_model_key(self):

        self.model: "RemoteableMixin"

        return self.model._remote_model_key()

    def remote_backend_postprocess_result(self, local_result: Dict[int, Graph]):

        from ...pydantics.Response import ResultModel

        return {
            id: ResultModel.from_graph(graph)
            for id, graph in local_result.items()
        }

    def remote_backend_handle_result_value(self, value: Dict[int, Dict[str, Any]]):

        for graph_id, saves in value.items():

            graph = self.bridge.id_to_graph[graph_id]

            for node_name, node_value in saves.items():
                graph.nodes[node_name]._value = node_value
                
            graph.alive = False
            
        self.bridge = None
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Dict, Iterable

from ...tracing.Bridge import Bridge
from ...tracing.Graph import Graph
from ...tracing.protocols import EarlyStopProtocol
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
        self,
        backend: Backend,
        model: "NNsight",
        *args,
        bridge: Bridge = None,
        **kwargs,
    ) -> None:

        self.bridge = Bridge() if bridge is None else bridge

        self.model = model

        GraphBasedContext.__init__(
            self,
            backend,
            bridge=self.bridge,
            proxy_class=self.model.proxy_class,
            *args,
            **kwargs,
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.model._session = None

        super().__exit__(exc_type, exc_val, exc_tb)

    def iter(self, iterable: Iterable, **kwargs) -> Iterator:
        """Creates an Iterator context to iteratively execute an intervention graph, with an update item at each iteration.

        Args:
            - iterable (Iterable): Data to iterate over.
            - return_context (bool): If True, returns the Iterator context. Default: False.

        Returns:
            Iterator: Iterator context.

        Example:
            Setup:
                .. code-block:: python
                    import torch
                    from collections import OrderedDict
                    input_size = 5
                    hidden_dims = 10
                    output_size = 2
                    model = nn.Sequential(OrderedDict([
                        ('layer1', torch.nn.Linear(input_size, hidden_dims)),
                        ('layer2', torch.nn.Linear(hidden_dims, output_size)),
                    ]))
                    input = torch.rand((1, input_size))

            Ex:
                .. code-block:: python
                    with model.session() as session:
                        l  = session.apply(list).save()
                        with session.iter([0, 1, 2]) as item:
                            l.append(item)
        """

        bridge = weakref.proxy(self.bridge)

        backend = BridgeBackend(bridge)

        return Iterator(
            iterable,
            backend,
            bridge=bridge,
            proxy_class=self.model.proxy_class,
            **kwargs,
        )

    ### BACKENDS ########

    def local_backend_execute(self) -> Dict[int, Graph]:

        try:
            super().local_backend_execute()
        except EarlyStopProtocol.EarlyStopException:
            pass

        local_result = self.bridge.id_to_graph

        self.bridge = weakref.proxy(self.bridge)

        return local_result

    def remote_backend_get_model_key(self) -> str:

        self.model: "RemoteableMixin"

        return self.model.to_model_key()

    def remote_backend_postprocess_result(self, local_result: Dict[int, Graph]):

        from ...schema.Response import ResultModel

        return {
            id: ResultModel.from_graph(graph)
            for id, graph in local_result.items()
        }

    def remote_backend_handle_result_value(
        self, value: Dict[int, Dict[str, Any]]
    ):

        for graph_id, saves in value.items():

            graph = self.bridge.id_to_graph[graph_id]

            for node_name, node_value in saves.items():
                graph.nodes[node_name]._value = node_value

            graph.alive = False

    def remote_backend_cleanup(self):

        self.bridge = weakref.proxy(self.bridge)

        graph = self.graph
        graph.alive = False

        if not isinstance(graph, weakref.ProxyType):
            self.graph = weakref.proxy(graph)

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__} at {hex(id(self))}&gt;"

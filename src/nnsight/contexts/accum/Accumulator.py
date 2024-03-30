from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from ...tracing import protocols
from ...tracing.Bridge import Bridge
from ...tracing.Graph import Graph
from ..backends import AccumulatorBackend, Backend, RemoteMixin
from .Collection import Collection
from .Iterator import Iterator

if TYPE_CHECKING:
    from ...models.NNsightModel import NNsight


class Accumulator(Collection, RemoteMixin):
    """An Accumulator is a root Collection that handles adding new Graphs and new Collections while accumulating.

    Attributes:
        bridge (Bridge): Bridge object which stores all Graphs added during accumulation and handles interaction between them
        graph (Graph): Root Graph where operations and values meant for access by all subsequent Graphs should be stored and referenced.
        model (NNsight): NNsight model this Accumulator is accumulating for.
        backend (Backend): Backend for this Accumulator object.
        collector_stack (List[Collection]): Stack of all Collections added during accumulation to keep track of which Collection to add a Tracer to when calling model.trace().
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

        Collection.__init__(self, AccumulatorBackend(self), self)

        self.backend = backend

    def __enter__(self) -> Accumulator:

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.model._accumulator = None

        if isinstance(exc_val, BaseException):
            raise exc_val

        self.backend(self)

    def iter(self, iterable) -> Iterator:

        backend = AccumulatorBackend(self)

        return Iterator(iterable, backend, self)

    ### BACKENDS ########

    def remote_backend_create_request(self):

        from ...pydantics.Request import RequestModel

        return RequestModel(object=self, repo_id=self.model._model_key)

    def remote_backend_handle_result(self, result: ResultModel) -> None:
        return super().remote_backend_handle_result(result)

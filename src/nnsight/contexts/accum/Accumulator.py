from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from nnsight.pydantics.Request import RequestModel
from nnsight.pydantics.Response import ResultModel

from ...tracing import protocols
from ...tracing.Bridge import Bridge
from ...tracing.Graph import Graph
from ..backends import AccumulatorBackend, Backend, LocalMixin, RemoteMixin
from .Collector import Collection
from .Iterator import Iterator

if TYPE_CHECKING:
    from ...models.NNsightModel import NNsight


class Accumulator(Collection, RemoteMixin):

    def __init__(self, backend: Backend, model: "NNsight") -> None:

        self.bridge = Bridge()
        self.graph = Graph(proxy_class=model.proxy_class)

        protocols.BridgeProtocol.set_bridge(self.graph, self.bridge)
        
        self.bridge.add(self.graph)

        Collection.__init__(self, AccumulatorBackend(self), self)

        self.model = model
        self.backend = backend

        self.collector_stack: List[Collection] = list()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(exc_val, BaseException):
            raise exc_val

        self.backend(self)

    def iter(self, iterable) -> Iterator:

        backend = AccumulatorBackend(self)

        return Iterator(iterable, backend, self)

    ### BACKENDS ########

    def remote_backend_create_request(self) -> RequestModel:
        return super().remote_backend_create_request()

    def remote_backend_handle_result(self, result: ResultModel) -> None:
        return super().remote_backend_handle_result(result)

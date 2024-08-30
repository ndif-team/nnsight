from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Tuple

from typing_extensions import Self

from ..tracing import protocols
from ..tracing.Bridge import Bridge
from ..tracing.Graph import Graph
from . import resolve_dependencies
from .backends import Backend, EditBackend, BridgeMixin, EditMixin, RemoteMixin
from .GraphBasedContext import GraphBasedContext
from .Invoker import Invoker

if TYPE_CHECKING:
    from ..models.mixins import RemoteableMixin
    from ..models.NNsightModel import NNsight


class Tracer(GraphBasedContext, RemoteMixin, BridgeMixin, EditMixin):
    """The Tracer class creates a :class:`nnsight.tracing.Graph.Graph` around the ._model of a :class:`nnsight.models.NNsightModel.NNsight` which tracks and manages the operations performed on the inputs and outputs of said model.

    Attributes:
        _model (nnsight.models.NNsightModel.NNsight): nnsight Model object that ths context manager traces and executes.
        _graph (nnsight.tracing.Graph.Graph): Graph which traces operations performed on the input and output of modules' Envoys are added and later executed.
        _args (List[Any]): Positional arguments to be passed to function that executes the model.
        _kwargs (Dict[str,Any]): Keyword arguments to be passed to function that executes the model.
        _invoker_inputs (List[Any]): Inputs for each invocation of this Tracer.
        _invoker (Invoker): Currently open Invoker.
    """

    def __init__(
        self,
        backend: Backend,
        model: "NNsight",
        validate: bool = False,
        graph: Graph = None,
        bridge: Bridge = None,
        return_context: bool = False,
        **kwargs,
    ) -> None:

        self.model = model

        self.return_context = return_context

        GraphBasedContext.__init__(
            self,
            backend,
            graph=graph,
            bridge=bridge,
            proxy_class=model.proxy_class,
            validate=validate,
            sequential=False,
        )

        protocols.ApplyModuleProtocol.set_module(self.graph, self.model)

        self._kwargs = kwargs

        self.invoker: Optional[Invoker] = None

        self._invoker_inputs: List[Any] = []

        # Module Envoys need to know about the current Tracer to create the correct proxies.
        self.model._envoy._set_tracer(weakref.proxy(self))

    def __getattr__(self, key: Any) -> Any:
        """Wrapper of .model._envoy's attributes to access module Envoy inputs and outputs.

        Returns:
            Any: Attribute.
        """
        return getattr(self.model._envoy, key)

    def __enter__(self) -> Union[Self, "NNsight", Tuple["NNsight", Self]]:

        tracer = super().__enter__()

        if self.invoker is not None:

            self.invoker.__enter__()

        if isinstance(self.backend, EditBackend):
            if self.return_context:
                return self.model, self
            
            return self.model

        return tracer

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        if self.invoker is not None:

            self.invoker.__exit__(None, None, None)

        self.model._envoy._reset()


        super().__exit__(exc_type, exc_val, exc_tb)

    def invoke(self, *inputs: Any, **kwargs) -> Invoker:
        """Create an Invoker context dor a given input.

        Raises:
            Exception: If an Invoker context is already open

        Returns:
            Invoker: Invoker.
        """

        if self.invoker is not None:

            raise Exception("Can't create an invoker context with one already open!")

        return Invoker(self, *inputs, **kwargs)

    def next(self, increment: int = 1) -> None:
        """Increments call_iter of all module Envoys. Useful when doing iterative/generative runs.

        Args:
            increment (int): How many call_iter to increment at once. Defaults to 1.
        """

        self.model._envoy.next(increment=increment, propagate=True)

    ##### BACKENDS ###############################

    def local_backend_execute(self) -> Graph:

        protocols.ApplyModuleProtocol.set_module(self.graph, self.model._model)

        self.graph.reset()

        invoker_inputs = self._invoker_inputs

        # If ths graph has a Bridge, we need to check for Nodes in the input itself.
        if protocols.BridgeProtocol.has_bridge(self.graph):

            invoker_inputs = resolve_dependencies(invoker_inputs)

        self.graph.execute()

        self.model.interleave(
            self.model._execute,
            self.graph,
            *invoker_inputs,
            **self._kwargs,
        )

        graph = self.graph
        graph.alive = False

        if not isinstance(graph, weakref.ProxyType):
            self.graph = weakref.proxy(graph)

        return graph

    def edit_backend_execute(self) -> Graph:

        self.model._default_graph = self.graph

    def remote_backend_get_model_key(self) -> str:

        self.model: "RemoteableMixin"

        return self.model.to_model_key()

    def remote_backend_postprocess_result(self, local_result: Graph) -> Dict[str, Any]:

        from ..schema.Response import ResultModel

        return ResultModel.from_graph(local_result)

    def remote_backend_handle_result_value(self, value: Dict[str, Any]) -> None:

        # TODO : graph mismatch handle. hash json ?
        for node_name, node_value in value.items():
            self.graph.nodes[node_name]._value = node_value

    def remote_backend_cleanup(self):
        
        graph = self.graph
        graph.alive = False

        if not isinstance(graph, weakref.ProxyType):
            self.graph = weakref.proxy(graph)

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__} at {hex(id(self))}&gt;"

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from typing_extensions import Self

from ..tracing import protocols
from ..tracing.Bridge import Bridge
from ..tracing.Graph import Graph
from . import resolve_dependencies
from .backends import Backend, BridgeMixin, EditBackend, EditMixin, RemoteMixin
from .GraphBasedContext import GraphBasedContext
from .Invoker import Invoker
from ..intervention import InterventionHandler
if TYPE_CHECKING:
    from ..models.mixins import RemoteableMixin
    from ..models.NNsightModel import NNsight
    from ..tracing.Node import Node


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
        graph: Optional[Graph] = None,
        bridge: Optional[Bridge] = None,
        method: Optional[str] = None,
        validate: bool = False,
        return_context: bool = False,
        **kwargs,
    ) -> None:

        self.model = model
        self.method = method

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
        """Create an Invoker context for a given input.

        Raises:
            Exception: If an Invoker context is already open

        Returns:
            Invoker: Invoker.
        """

        if self.invoker is not None:

            raise Exception("Can't create an invoker context with one already open!")

        return Invoker(self, *inputs, **kwargs)

    def batch(
        self, invoker_inputs: Tuple[Tuple[Tuple[Any], Dict[str, Any]]]
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], List[Tuple[int, int]]]:

        batch_groups = []
        batch_start = 0
        batched_input = None

        for args, kwargs in invoker_inputs:
            (args, kwargs), batch_size = self.model._prepare_input(*args, **kwargs)

            batch_groups.append((batch_start, batch_size))

            batched_input = self.model._batch(batched_input, *args, **kwargs)

            batch_start += batch_size

        if batched_input is None:
            
            batched_input = (((0, -1),), dict())

        return batched_input, batch_groups
    
    @property
    def _invoker_group(self):
        
        return len(self._invoker_inputs) - 1

    ##### BACKENDS ###############################

    def local_backend_execute(self) -> Graph:

        protocols.ApplyModuleProtocol.set_module(self.graph, self.model._model)

        self.graph.reset()

        invoker_inputs = self._invoker_inputs

        # If ths graph has a Bridge, we need to check for Nodes in the input itself.
        if protocols.BridgeProtocol.has_bridge(self.graph):

            invoker_inputs = resolve_dependencies(invoker_inputs)
            
        (args, kwargs), batch_groups = self.batch(invoker_inputs)

        self.graph.execute()

        fn = (
            self.model._execute
            if self.method is None
            else getattr(self.model, self.method)
        )
        
        intervention_handler = InterventionHandler(batch_groups=batch_groups)

        self.model.interleave(
            fn,
            self.graph,
            *args,
            intervention_handler=intervention_handler,
            **kwargs,
            **self._kwargs,
        )

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

    def remote_backend_get_stream_node(self, name: str, graph_id: str) -> "Node":
        return self.graph.nodes[name]

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__} at {hex(id(self))}&gt;"

from __future__ import annotations

import inspect
import weakref
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

from .. import pydantics, util
from ..intervention import InterventionProxy
from ..tracing import protocols
from ..tracing.Graph import Graph
from ..tracing.Node import Node
from .backends import (AccumulatorMixin, Backend, IteratorMixin, LocalMixin,
                       RemoteMixin)
from .Invoker import Invoker

if TYPE_CHECKING:
    from ..models.NNsightModel import NNsight
    from ..pydantics.Request import RequestModel
    from ..pydantics.Response import ResultModel
    from .accum.Accumulator import Accumulator


class Tracer(
    AbstractContextManager, LocalMixin, RemoteMixin, AccumulatorMixin, IteratorMixin
):
    """The Tracer class creates a :class:`nnsight.tracing.Graph.Graph` around the ._model of a :class:`nnsight.models.NNsightModel.NNsight` which tracks and manages the operations performed on the inputs and outputs of said model.

    Attributes:
        _model (nnsight.models.NNsightModel.NNsight): nnsight Model object that ths context manager traces and executes.
        _graph (nnsight.tracing.Graph.Graph): Graph which traces operations performed on the input and output of modules' Envoys are added and later executed.
        _args (List[Any]): Positional arguments to be passed to function that executes the model.
        _kwargs (Dict[str,Any]): Keyword arguments to be passed to function that executes the model.
        _batch_size (int): Batch size of the most recent input. Used by Envoy to create input/output proxies.
        _batch_start (int): Batch start of the most recent input. Used by Envoy to create input/output proxies.
        _batched_input (Any): Batched version of all inputs involved in this Tracer.
        _invoker (Invoker): Currently open Invoker.
    """

    def __init__(
        self,
        backend: Backend,
        model: "NNsight",
        graph: Graph = None,
        validate: bool = False,
        **kwargs,
    ) -> None:

        self._model = model

        self._graph = (
            Graph(proxy_class=model.proxy_class, validate=validate)
            if graph is None
            else graph
        )

        protocols.ApplyModuleProtocol.set_module(self._graph, self._model)

        self._backend = backend

        self._kwargs = kwargs

        self._invoker: Invoker = None

        self._batch_size: int = 0
        self._batch_start: int = 0

        self._batched_input: Any = None

        # Module Envoys need to know about the current Tracer to create the correct proxies.
        self._model._envoy._set_tracer(weakref.proxy(self))

    def __getattr__(self, key: Any) -> Any:
        """Wrapper of .model._envoy's attributes to access module Envoy inputs and outputs.

        Returns:
            Any: Attribute.
        """
        return getattr(self._model._envoy, key)

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(exc_val, BaseException):
            raise exc_val

        self._backend(self)

    def invoke(self, *inputs: Tuple[Any], **kwargs) -> Invoker:
        """Create an Invoker context dor a given input.

        Raises:
            Exception: If an Invoker context is already open

        Returns:
            Invoker: Invoker.
        """

        if self._invoker is not None:

            raise Exception("Can't create an invoker context with one already open!")

        return Invoker(self, *inputs, **kwargs)

    def next(self, increment: int = 1) -> None:
        """Increments call_iter of all module Envoys. Useful when doing iterative/generative runs.

        Args:
            increment (int): How many call_iter to increment at once. Defaults to 1.
        """

        self._model._envoy.next(increment=increment, propagate=True)

    def apply(
        self, target: Callable, validate: bool = False, *args, **kwargs
    ) -> InterventionProxy:
        """Helper method to directly add a function to the intervention graph.

        Args:
            target (Callable): Function to apply
            validate (bool): If to try and run this operation in FakeMode to test it out and scan it.

        Returns:
            InterventionProxy: Proxy of applying that function.
        """
        return self._graph.add(
            target=target,
            value=inspect._empty if validate else None,
            args=args,
            kwargs=kwargs,
        )

    ##### BACKENDS ###############################

    def local_backend_execute(self):

        self._graph.compile()

        protocols.ApplyModuleProtocol.set_module(self._graph, self._model._model)

        _batched_input = self._batched_input

        # If ths graph has a Bridge, we need to check for Nodes in the input itself.
        if protocols.BridgeProtocol.has_bridge(self._graph):

            def get_value(node: Node):

                value = node.args[0].value

                node.set_value(None)

                return value

            _batched_input = util.apply(_batched_input, get_value, Node)

        self._model.interleave(
            self._model._execute,
            self._graph,
            *_batched_input,
            **self._kwargs,
        )

        self._graph.alive = False
        self._graph = None

    def remote_backend_create_request(self) -> "RequestModel":

        from ..pydantics.Request import RequestModel

        return RequestModel(object=self, repo_id=self._model._model_key)

    def remote_backend_create_result(
        self,
    ) -> "ResultModel":

        # TODO

        from ..pydantics.Response import ResultModel

        saves = ResultModel.from_graph(self._graph)

        return ResultModel(saves=[saves])

    def remote_backend_handle_result(self, result: pydantics.ResultModel) -> None:

        # Set save data.
        # TODO
        for name, value in result.saves.items():
            self._graph.nodes[name]._value = value

        self._graph.alive = False
        self._graph = None

    def accumulator_backend_handle(self, accumulator: "Accumulator") -> None:

        accumulator.collector_stack[-1].collection.append(self)

        protocols.BridgeProtocol.set_bridge(self._graph, accumulator.bridge)

        accumulator.bridge.add(self._graph)

    def iterator_backend_execute(self, last_iter: bool = False) -> None:

        graph = self._graph

        self.local_backend_execute()

        if not last_iter:

            self._graph = graph
            self._graph.alive = True

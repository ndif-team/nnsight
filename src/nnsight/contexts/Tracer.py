from __future__ import annotations

import weakref
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

from nnsight.pydantics import RequestModel

from .. import pydantics
from ..intervention import InterventionProxy
from ..tracing import protocols
from ..tracing.Graph import Graph
from .backends import Backend, LocalMixin, RemoteMixin
from .Invoker import Invoker

if TYPE_CHECKING:
    from ..models.NNsightModel import NNsight


class Executable(LocalMixin):

    def __init__(self, graph: Graph = None) -> None:

        if graph is None:
            graph = Graph()

        self.executable_graph: Graph = graph


class Tracer(AbstractContextManager, Executable, RemoteMixin):
    """The Tracer class creates a :class:`nnsight.tracing.Graph.Graph` around the ._model of a :class:`nnsight.models.NNsightModel.NNsight` which tracks and manages the operations performed on the inputs and outputs of said model.

    Attributes:
        model (nnsight.models.NNsightModel.NNsight): nnsight Model object that ths context manager traces and executes.
        graph (nnsight.tracing.Graph.Graph): Graph which traces operations performed on the input and output of modules' Envoys are added and later executed.
        args (List[Any]): Positional arguments to be passed to function that executes the model.
        kwargs (Dict[str,Any]): Keyword arguments to be passed to function that executes the model.
        batch_size (int): Batch size of the most recent input. Used by Envoy to create input/output proxies.
        batch_start (int): Batch start of the most recent input. Used by Envoy to create input/output proxies.
        batched_input Any: Batched version of all inputs involved in this Tracer.
    """

    def __init__(
        self,
        backend: Backend,
        model: "NNsight",
        validate: bool = False,
        **kwargs,
    ) -> None:

        self._model = model

        graph = Graph(proxy_class=model.proxy_class, validate=validate)

        Executable.__init__(self, graph)

        protocols.ApplyModuleProtocol.set_module(self.executable_graph, self._model)

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

        self.executable_graph.tracing = False
        self.executable_graph = None

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

    def apply(self, target: Callable, *args, **kwargs) -> InterventionProxy:
        """Helper method to directly add a function to the intervention graph.

        Args:
            target (Callable): Function to apply

        Returns:
            InterventionProxy: Proxy of applying that function.
        """
        return self.executable_graph.add(target=target, args=args, kwargs=kwargs)

    ##### BACKENDS ###############################

    def local_backend_execute(self):

        self.executable_graph.compile()
        
        protocols.ApplyModuleProtocol.set_module(self.executable_graph, self._model._model)

        self._model.interleave(
            self._model._execute,
            self.executable_graph,
            *self._batched_input,
            **self._kwargs,
        )

    def remote_backend_create_request(self) -> RequestModel:

        return pydantics.RequestModel(
            kwargs=self._kwargs,
            repo_id=self._model._model_key,
            batched_input=self._batched_input,
            intervention_graph=self.executable_graph.nodes,
        )

    def remote_backend_handle_result(self, result: pydantics.ResultModel) -> None:

        # Set save data.
        for name, value in result.saves.items():
            self.executable_graph.nodes[name].value = value

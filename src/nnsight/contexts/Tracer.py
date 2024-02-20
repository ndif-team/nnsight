from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

from ..intervention import InterventionProxy
from ..tracing.Graph import Graph
from .Invoker import Invoker

if TYPE_CHECKING:
    from ..models.NNsightModel import NNsight


class Tracer:
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
        model: "NNsight",
        validate: bool = True,
        **kwargs,
    ) -> None:

        self._model = model

        self._kwargs = kwargs

        self._graph = Graph(
            self._model._envoy, proxy_class=model.proxy_class, validate=validate
        )

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
        
        output = self._model.interleave(
            self._model._execute,
            self._graph,
            *self._batched_input,
            **self._kwargs,
        )

        self._graph = None

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
        return self._graph.add(target=target, args=args, kwargs=kwargs)

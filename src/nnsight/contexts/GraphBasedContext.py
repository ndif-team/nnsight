from __future__ import annotations

import inspect
import weakref
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable

from torch.utils._python_dispatch import TorchDispatchMode
from typing_extensions import Self

from ..intervention import InterventionProxy
from ..tracing import protocols
from ..tracing.Bridge import Bridge
from ..tracing.Graph import Graph
from .backends import Backend, BridgeMixin


class GraphBasedContext(AbstractContextManager, BridgeMixin):

    def __init__(
        self,
        backend: Backend,
        graph: Graph = None,
        bridge: Bridge = None,
        **kwargs,
    ) -> None:

        self.backend = backend

        self.graph: Graph = Graph(**kwargs) if graph is None else graph

        if bridge is not None:

            bridge.add(self.graph)

    def apply(
        self,
        target: Callable,
        *args,
        validate: bool = False,
        **kwargs,
    ) -> InterventionProxy:
        """Helper method to directly add a function to the intervention graph.

        Args:
            target (Callable): Function to apply
            validate (bool): If to try and run this operation in FakeMode to test it out and scan it.

        Returns:
            InterventionProxy: Proxy of applying that function.
        """
        return self.graph.create(
            target=target,
            proxy_value=inspect._empty if validate else None,
            args=args,
            kwargs=kwargs,
        )

    def exit(self) -> InterventionProxy:
        """Exits the execution of a sequential intervention graph.

        Returns:
            InterventionProxy: Proxy of the EarlyStopProtocol node.
        """

        if self.graph.sequential:
            return protocols.EarlyStopProtocol.add(self.graph)
        else:
            raise Exception(
                "Early exit is only supported for sequential graph-based contexts."
            )

    def log(self, *data: Any) -> None:
        """Adds a node via .apply to print the value of a Node.

        Args:
            data (Any): Data to print.
        """
        self.apply(print, *data)

    def vis(self, **kwargs) -> None:
        """
        Helper method to save a visualization of the current state of the intervention graph.
        """

        self.graph.vis(**kwargs)

    def __enter__(self) -> Self:

        GlobalTracingContext.try_register(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        GlobalTracingContext.try_deregister(self)

        if isinstance(exc_val, BaseException):
            self.graph.alive = False
            self.graph = None
            raise exc_val

        self.backend(self)

    ### BACKENDS ########

    def local_backend_execute(self) -> None:

        try:
            self.graph.reset()
            self.graph.execute()
        except protocols.EarlyStopProtocol.EarlyStopException as e:
            raise e
        finally:
            graph = self.graph
            graph.alive = False

            if not isinstance(graph, weakref.ProxyType):
                self.graph = weakref.proxy(graph)

    def bridge_backend_handle(self, bridge: Bridge) -> None:

        bridge.pop_graph()

        protocols.LocalBackendExecuteProtocol.add(self, bridge.peek_graph())

        self.graph = weakref.proxy(self.graph)


class GlobalTracingContext(GraphBasedContext):
    """The Global Tracing Context handles adding tracing operations globally without reference to a given `GraphBasedContext`.
    There should only be one of these and that is `GlobalTracingContext.GLOBAL_TRACING_CONTEXT`.
    `GlobalTracingContext.TORCH_DISPATCHER` handles adding torch functions without reference to a given `GraphBasedContext`.

    """

    GLOBAL_TRACING_CONTEXT: GlobalTracingContext
    TORCH_DISPATCHER: GlobalTracingContext.GlobalTracingDispatcher

    class GlobalTracingDispatcher(TorchDispatchMode):

        def __torch_dispatch__(self, func, types, args, kwargs=None):

            return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(
                func, *args, **kwargs
            )

    def __init__(self) -> None:
        """We create an empty `GraphBasedContext` by default."""

        self.graph: Graph = None

    @staticmethod
    def try_register(graph_based_context: GraphBasedContext) -> bool:
        """Attempts to register a `Graph` globally.]
        Will not if one is already registered.

        Args:
            graph_based_context (GraphBasedContext): `GraphBasedContext` to register.

        Returns:
            bool: True if registering ws successful, False otherwise.
        """

        if GlobalTracingContext.GLOBAL_TRACING_CONTEXT:

            return False

        GlobalTracingContext.register(graph_based_context)

        return True

    @staticmethod
    def try_deregister(graph_based_context: GraphBasedContext) -> bool:
        """Attempts to deregister a `Graph` globally.
        Will not if `graph_based_context` does not have the same `Graph` as the currently registered one.

        Args:
            graph_based_context (GraphBasedContext): `GraphBasedContext` to deregister.

        Returns:
            bool: True if deregistering ws successful, False otherwise.
        """
        if (
            not GlobalTracingContext.GLOBAL_TRACING_CONTEXT
            or graph_based_context.graph
            is not GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph
        ):

            return False
        GlobalTracingContext.deregister()

        return True

    @staticmethod
    def register(graph_based_context: GraphBasedContext) -> None:
        """Register `GraphBasedContext` globally.

        Args:
            graph_based_context (GraphBasedContext): GraphBasedContext to register.
        """

        assert GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is None

        GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph = graph_based_context.graph

        GlobalTracingContext.TORCH_DISPATCHER.__enter__()

    @staticmethod
    def deregister() -> None:
        """Deregister `GraphBasedContext` globally.

        Args:
            graph_based_context (GraphBasedContext): GraphBasedContext to deregister.
        """

        assert GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is not None

        GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph = None

        GlobalTracingContext.TORCH_DISPATCHER.__exit__(None, None, None)

    def __bool__(self) -> bool:
        """True if there is a `GraphBasedContext` registered globally. False otherwise."""

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is not None

    def __getattribute__(self, name: str) -> Any:
        """Prevent attribute access if no `GraphBasedContext` registered."""

        static_methods = [
            name
            for name, value in inspect.getmembers(
                GraphBasedContext, predicate=inspect.ismethod
            )
        ]

        if name in static_methods:

            if not GlobalTracingContext.GLOBAL_TRACING_CONTEXT:

                raise Exception(
                    "Global ops cannot be used outside of a tracing context."
                )

        return object.__getattribute__(self, name)


GlobalTracingContext.GLOBAL_TRACING_CONTEXT = GlobalTracingContext()
GlobalTracingContext.TORCH_DISPATCHER = GlobalTracingContext.GlobalTracingDispatcher()

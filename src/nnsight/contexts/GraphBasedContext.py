from __future__ import annotations

import inspect
import weakref
from contextlib import AbstractContextManager
from functools import wraps
from typing import Any, Callable, Union

import torch
from torch.overrides import TorchFunctionMode
from typing_extensions import Self

from ..intervention import InterventionProxy
from ..patching import Patch, Patcher
from ..tracing import protocols
from ..tracing.Bridge import Bridge
from ..tracing.Graph import Graph
from .backends import Backend, BridgeMixin
from .Conditional import Conditional


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
        validate: bool = None,
        **kwargs,
    ) -> InterventionProxy:
        """Helper method to directly add a function to the intervention graph.

        Args:
            target (Callable): Function to apply
            validate (bool): If to try and run this operation in FakeMode to test it out and scan it.

        Returns:
            InterventionProxy: Proxy of applying that function.
        """

        proxy_value = inspect._empty

        if validate is False:

            proxy_value = None

        return self.graph.create(
            target=target,
            proxy_value=proxy_value,
            args=args,
            kwargs=kwargs,
        )

    def cond(self, condition: Union[InterventionProxy, Any]) -> Conditional:
        """Entrypoint to the Conditional context.
            Takes in a condition argument which acts as the dependency of the Conditional node in the Intervention graph.
            The condition is evaluated as a boolean, and if True, executes all the interventions defined within the body
            of the conditional context.

        Args:
            condition (Union[InterventionProxy, Any]): Dependency of the Conditional Node.

        Returns:
            Conditional: Conditional context object.

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

            Ex 1: The .save() on the model output will only be executed if the condition passed to tracer.cond() is evaluated to True.

            .. code-block:: python
                x: int = 5
                with model.trace(input) as trace:
                    with tracer.cond(x > 0):
                        out = model.output.save()

            Ex 2: The condition is on an InterventionProxy which creates in return an InterventionProxy

            .. code-block:: python
                with model.trace(input) as trace:
                    with tracer.cond(model.layer1.output[:, 0] > 0):
                        out = model.output.save()
        """

        return Conditional(self.graph, condition)

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

    def bool(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable bool."""

        return self.apply(bool, *args, **kwargs)

    def bytes(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable bytes."""

        return self.apply(bytes, *args, **kwargs)

    def int(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable int."""

        return self.apply(int, *args, **kwargs)

    def float(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable float."""

        return self.apply(float, *args, **kwargs)

    def str(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable string."""

        return self.apply(str, *args, **kwargs)

    def complex(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable complex number."""

        return self.apply(complex, *args, **kwargs)

    def bytearray(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable bytearray."""

        return self.apply(bytearray, *args, **kwargs)

    def tuple(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable tuple."""

        return self.apply(tuple, *args, **kwargs)

    def list(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable list."""

        return self.apply(list, *args, **kwargs)

    def set(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable set."""

        return self.apply(set, *args, **kwargs)

    def dict(self, *args, **kwargs) -> InterventionProxy:
        """NNsight helper method to create a traceable dictionary."""

        return self.apply(dict, *args, **kwargs)

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


from inspect import getmembers, isclass

from torch.utils import data


def global_patch(root, name: str) -> Patch:

    fn = getattr(root, name)

    @wraps(fn)
    def inner(*args, **kwargs):

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(
            fn, *args, **kwargs
        )

    return Patch(root, inner, name)


def global_patch_class(cls: type) -> Patch:

    if cls.__new__ is object.__new__:

        def super_new(cls, *args, **kwargs):

            return object.__new__(cls)

        cls.__new__ = super_new

    fn = cls.__new__

    @wraps(fn)
    def inner(cls, *args, **kwargs):

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(
            cls, *args, **kwargs
        )

    return Patch(cls, inner, "__new__")


class GlobalTracingContext(GraphBasedContext):
    """The Global Tracing Context handles adding tracing operations globally without reference to a given `GraphBasedContext`.
    There should only be one of these and that is `GlobalTracingContext.GLOBAL_TRACING_CONTEXT`.
    `GlobalTracingContext.TORCH_HANDLER` handles adding torch functions without reference to a given `GraphBasedContext`.

    """

    GLOBAL_TRACING_CONTEXT: GlobalTracingContext
    TORCH_HANDLER: GlobalTracingContext.GlobalTracingTorchHandler
    PATCHER: Patcher = Patcher(
        [
            global_patch_class(torch.nn.Parameter),
            global_patch_class(data.DataLoader),
            global_patch(torch, "arange"),
            global_patch(torch, "empty"),
            global_patch(torch, "eye"),
            global_patch(torch, "full"),
            global_patch(torch, "linspace"),
            global_patch(torch, "logspace"),
            global_patch(torch, "ones"),
            global_patch(torch, "rand"),
            global_patch(torch, "randint"),
            global_patch(torch, "randn"),
            global_patch(torch, "randperm"),
            global_patch(torch, "zeros"),
            global_patch(torch, "cat")
        ]
        + [
            global_patch_class(value)
            for key, value in getmembers(torch.optim, isclass)
            if issubclass(value, torch.optim.Optimizer)
        ]
    )

    class GlobalTracingTorchHandler(TorchFunctionMode):

        def __torch_function__(self, func, types, args, kwargs=None):

            if kwargs is None:

                kwargs = {}

            if "_VariableFunctionsClass" in func.__qualname__:
                return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(
                    func, *args, **kwargs
                )

            return func(*args, **kwargs)

    class GlobalTracingExit(AbstractContextManager):

        def __enter__(self) -> Any:

            GlobalTracingContext.TORCH_HANDLER.__exit__(None, None, None)
            GlobalTracingContext.PATCHER.__exit__(None, None, None)

            return self

        def __exit__(self, exc_type, exc_val, traceback):

            GlobalTracingContext.TORCH_HANDLER.__enter__()
            GlobalTracingContext.PATCHER.__enter__()

            if isinstance(exc_val, BaseException):

                raise exc_val

    def __init__(self) -> None:
        """We create an empty `GraphBasedContext` by default."""

        self.graph: Graph = None

    @staticmethod
    def exit_global_tracing_context():

        return GlobalTracingContext.GlobalTracingExit()

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

        GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph = (
            graph_based_context.graph
        )

        GlobalTracingContext.TORCH_HANDLER.__enter__()
        GlobalTracingContext.PATCHER.__enter__()

    @staticmethod
    def deregister() -> None:
        """Deregister `GraphBasedContext` globally.

        Args:
            graph_based_context (GraphBasedContext): GraphBasedContext to deregister.
        """

        assert GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is not None

        GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph = None

        GlobalTracingContext.TORCH_HANDLER.__exit__(None, None, None)
        GlobalTracingContext.PATCHER.__exit__(None, None, None)

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
GlobalTracingContext.TORCH_HANDLER = (
    GlobalTracingContext.GlobalTracingTorchHandler()
)

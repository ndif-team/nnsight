from __future__ import annotations

import inspect
import weakref
from contextlib import AbstractContextManager
from typing import Any, Callable, Union

from typing_extensions import Self

from ..intervention import InterventionProxy
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

    def cond(self, condition: Union[InterventionProxy, Any]) -> Conditional:
        """ Entrypoint to the Conditional context. 
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

    def vis(self, **kwargs) -> None:
        """
        Helper method to save a visualization of the current state of the intervention graph.
        """

        self.graph.vis(**kwargs)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(exc_val, BaseException):
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

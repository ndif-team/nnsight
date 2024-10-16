from __future__ import annotations

import inspect
import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Union

import torch

from ... import util
from .. import protocols
from .proxy import Proxy

if TYPE_CHECKING:
    from .graph import Graph


class Node:

    def __init__(
        self,
        target: Union[Callable, protocols.Protocol],
        *args,
        graph: "Graph" = None,
        trace_value: Any = inspect._empty,
        **kwargs,
    ) -> None:

        self.index: Optional[int] = None

        args = list(args)

        self.graph: "Graph" = graph

        self.target = target

        self.args = args
        self.kwargs = kwargs

        self.fake_value = trace_value

        self.listeners: List[Node] = list()
        self.dependencies: List[Node] = list()
        self.condition: Optional[Node] = None

        self._value: Any = inspect._empty
        self.remaining_listeners = 0
        self.remaining_dependencies = 0
        self.executed = False

        # Preprocess args.
        self.preprocess()

        # Node.graph is a weak reference to avoid reference loops.
        self.graph = weakref.proxy(self.graph) if self.graph is not None and not isinstance(graph, weakref.ProxyType) else None

        # If theres an alive Graph, add it.
        if self.attached:

            self.graph.add(self)

    def preprocess(self) -> None:
        """Preprocess Node.args and Node.kwargs."""

        def preprocess_node(node: Union[Node, Proxy]):

            if isinstance(node, Proxy):

                node = node.node

            if node.done:

                return node.value

            self.dependencies.append(node)
            # Weakref so no reference loop
            node.listeners.append(weakref.proxy(self))

            return node

        self.args, self.kwargs = util.apply(
            (self.args, self.kwargs), preprocess_node, (Node, Proxy)
        )

    ### Properties ########################
    @property
    def value(self) -> Any:
        """Property to return the value of this node.

        Returns:
            Any: The stored value of the node, populated during execution of the model.

        Raises:
            ValueError: If the underlying ._value is inspect._empty (therefore never set or destroyed).
        """

        if not self.done:
            raise ValueError("Accessing value before it's been set.")

        return self._value

    @property
    def attached(self) -> bool:
        """Checks to see if the weakref to the Graph is alive or dead.

        Returns:
            bool: Is Node attached.
        """

        try:

            return self.graph.alive

        except:
            return False

    @property
    def done(self) -> bool:
        """Returns true if the value of this node has been set.

        Returns:
            bool: If done.
        """
        return self._value is not inspect._empty

    @property
    def fulfilled(self) -> bool:
        """Returns true if remaining_dependencies is 0.

        Returns:
            bool: If fulfilled.
        """
        return self.remaining_dependencies == 0

    @property
    def redundant(self) -> bool:
        """Returns true if remaining_listeners is 0.

        Returns:
            bool: If redundant.
        """
        return self.remaining_listeners == 0

    ### API #############################
    def reset(self) -> None:
        """Resets this Nodes remaining_listeners and remaining_dependencies."""

        self.executed = False
        self._value = inspect._empty

        self.remaining_listeners = len(self.listeners)
        self.remaining_dependencies = sum(
            [not node.executed for node in self.dependencies]
        ) + int(not (self.condition is None))

    def create(
        self,
        target: Union[Callable, str],

        *args,
        **kwargs,
    ) -> Union[Proxy, Any]:
        """We use Node.add vs Graph.add in case graph is dead.
        If the graph is dead, we assume this node is ready to execute and therefore we try and execute it and then return its value.

        Returns:
            Union[Proxy, Any]: Proxy or value
        """

        if not self.attached:

            from ..contexts import GlobalTracingContext

            if GlobalTracingContext.GLOBAL_TRACING_CONTEXT:

                return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph.create(
                    target=target,
                    proxy_value=proxy_value,
                    args=args,
                    kwargs=kwargs,
                )

            # Create Node with no values or Graph.
            node = Node(
                target,
                *args,
                trace_value=None,
                **kwargs,
            )

            # Reset it.
            node.reset()

            # So it doesn't get destroyed.
            node.remaining_listeners = 1

            # Execute Node
            node.execute()

            # Get value.
            value = node.value

            # Destroy.
            node.destroy()

            return value
        # Otherwise just create the Node on the Graph like normal.
        return self.graph.create(
            target,
            *args,
            **kwargs,
        )

    @classmethod
    def prepare_inputs(
        cls, inputs: Any, device: torch.device = None, proxy: bool = False
    ) -> Any:
        """Prepare arguments for executing this node's target.
        Converts Nodes in args and kwargs to their value and moves tensors to correct device.

        Returns:
            Any: Prepared inputs.
        """

        inputs = util.apply(inputs, lambda x: x, inspect._empty)

        def _value(node: Proxy | Node):

            if isinstance(node, Proxy):
                node = node.node

            if proxy:
                return node.fake_value

            return node.value

        inputs = util.apply(inputs, _value, (Node, Proxy), inplace=not proxy)

        if device is None:

            def _device(value: torch.Tensor):
                nonlocal device

                if device is None:
                    device = value.device

            util.apply(inputs, _device, torch.Tensor)

        def _to(value: torch.Tensor):
            return value.to(device)

        inputs = util.apply(inputs, _to, torch.Tensor, inplace=not proxy)

        return inputs

    def execute(self) -> None:
        """Actually executes this node.
        Lets protocol execute if target is str.
        Else prepares args and kwargs and passes them to target. Gets output of target and sets the Node's value to it.
        """

        self.executed = True

        try:

            if isinstance(self.target, type) and issubclass(
                self.target, protocols.Protocol
            ):

                self.target.execute(self)

            else:

                # Prepare arguments.
                args, kwargs = Node.prepare_inputs((self.args, self.kwargs))

                # Call the target to get value.
                output = self.target(*args, **kwargs)

                # Set value.
                self.set_value(output)

        except Exception as e:

            raise e

    def set_value(self, value: Any) -> None:
        """Sets the value of this Node and logs the event.
        Updates remaining_dependencies of listeners. If they are now fulfilled, execute them.
        Updates remaining_listeners of dependencies. If they are now redundant, destroy them.

        Args:
            value (Any): Value.
        """
        self._value = value

        self.update_listeners()

        self.update_dependencies()

        if self.done and self.redundant:
            self.destroy()

    def update_listeners(self):
        """Updates remaining_dependencies of listeners."""

        for listener in self.listeners:
            listener.remaining_dependencies -= 1

    def update_dependencies(self):
        """Updates remaining_listeners of dependencies. If they are now redundant, destroy them."""

        for dependency in self.dependencies:
            dependency.remaining_listeners -= 1

            if dependency.redundant:
                dependency.destroy()

    def destroy(self) -> None:
        """Removes the reference to the node's value and logs it's destruction."""

        self._value = inspect._empty

    def subgraph(self, subgraph: Optional[Set[int]] = None) -> Set[int]:

        if subgraph is None:
            subgraph = set()

        if self.index in subgraph:
            return subgraph

        subgraph.add(self.index)

        for listener in self.listeners:
            listener.subgraph(subgraph)

        return subgraph

    def clean(self) -> None:
        """Clean up dependencies during early execution stop"""

        for dependency in self.dependencies:
            dependency.remaining_listeners -= 1
            if dependency.redundant:
                dependency.destroy()

    ### Magic Methods #####################################
    def __str__(self) -> str:
        return f"{self.target.__name__} {self.index}"

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__} at {hex(id(self))}&gt;"

    def __hash__(self) -> int:
        return id(self)

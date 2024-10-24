from __future__ import annotations

import inspect
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)
from typing_extensions import Self
from ... import util
from ..protocols import Protocol
from .proxy import Proxy, ProxyType

if TYPE_CHECKING:
    from .graph import Graph


class Node:

    def __init__(
        self,
        target: Union[Callable, Protocol],
        *args,
        graph: "Graph" = None,
        **kwargs,
    ) -> None:

        self.index: Optional[int] = None

        # No tuples. Only lists.
        args = list(args)

        self.graph: "Graph" = graph

        self.target = target

        self.args = args
        self.kwargs = kwargs

        self._listeners: Set[int] = set()
        self._dependencies: Set[int] = set()

        self._value: Any = inspect._empty
        self.remaining_listeners = 0
        self.remaining_dependencies = 0
        self.executed = False

        # If theres an alive Graph, add it.
        if self.attached:

            self.graph.add(self)
            
            # Preprocess args.
            self.preprocess()
            
    @property
    def listeners(self) -> List[Self]:
        
        return [self.graph.nodes[index] for index in self._listeners]
    
    @property
    def dependencies(self) -> List[Self]:
        
        return [self.graph.nodes[index] for index in self._dependencies]

    def preprocess(self) -> None:
        """Preprocess Node.args and Node.kwargs.
        Converts Proxies to their Node.
        Converts Nodes that are done to their value.
        Adds Node arguments to self dependencies.
        Add self to Node argument listeners.
        """

        def preprocess_node(node: Union[NodeType, ProxyType]):

            if isinstance(node, Proxy):

                node = node.node

            if node.done:

                return node.value

            self._dependencies.add(node.index)
            node._listeners.add(self.index)

            return node

        self.args, self.kwargs = util.apply(
            (self.args, self.kwargs), preprocess_node, (Node, Proxy)
        )

    ### Properties ########################
    @property
    def value(self) -> Any:
        """Property to return the value of this node.

        Returns:
            Any: The stored value of the node, populated during execution.

        Raises:
            ValueError: If the underlying ._value is inspect._empty (therefore never set or was destroyed).
        """

        if not self.done:
            raise ValueError("Accessing value before it's been set.")

        return self._value

    @property
    def attached(self) -> bool:
        """Checks to see if the weakref to the Graph is alive or dead.
        Alive meaning the Graph is still open to tracing new Nodes.

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

        self.remaining_listeners = len(self._listeners)
        self.remaining_dependencies = sum(
            [not node.executed for node in self.dependencies]
        )

    def create(
        self,
        *args,
        **kwargs,
    ) -> Union[NodeType, Any]:
        """We use Node.create vs Graph.create in case graph is dead.
        If the graph is dead, we first check the GlobalTracing Context to add
        assume this node is ready to execute and therefore we try and execute it and then return its value.

        Returns:
            Union[NodeType, Any]: Proxy or value
        """

        if not self.attached:

            from ..contexts.globals import GlobalTracingContext

            if GlobalTracingContext.GLOBAL_TRACING_CONTEXT:

                return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph.create(
                    *args,
                    **kwargs,
                )

            # Create dangling Node.
            node = type(self)(
                *args,
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
            *args,
            **kwargs,
        )

    @classmethod
    def prepare_inputs(cls, inputs: Any) -> Any:
        """Prepare arguments for executing this node's target.
        Converts Nodes in args and kwargs to their value.

        Returns:
            Any: Prepared inputs.
        """

        inputs = util.apply(inputs, lambda x: x, inspect._empty)

        def _value(node: Union[ProxyType, NodeType]):

            if isinstance(node, Proxy):
                node = node.node

            return node.value

        inputs = util.apply(inputs, _value, (Node, Proxy))

        return inputs

    def execute(self) -> None:
        """Actually executes this node.
        Lets protocol execute if target is Protocol.
        Else prepares args and kwargs and passes them to target. Gets output of target and sets the Node's value to it.
        """

        self.executed = True

        try:

            if isinstance(self.target, type) and issubclass(self.target, Protocol):

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
        
        if self.graph is not None:

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
            if len(self.graph.defer_stack) > 0 and dependency.index  < self.graph.defer_stack[-1]:
                continue
            
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
    ### Magic Methods #####################################
    def __str__(self) -> str:
        return f"{self.target.__name__} {self.index}"

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__} at {hex(id(self))}&gt;"

    def __hash__(self) -> int:
        return id(self)


NodeType = TypeVar("NodeType", bound=Node)
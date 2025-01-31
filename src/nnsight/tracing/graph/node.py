from __future__ import annotations

import inspect
import re
import traceback
from typing import (TYPE_CHECKING, Any, Callable, Dict, List,
                    Optional, Set, TypeVar, Union)

from typing_extensions import Self

from ... import util
from ..protocols import Protocol
from .proxy import Proxy, ProxyType

from ...util import NNsightError

if TYPE_CHECKING:
    from .graph import Graph


class Node:
    """A computation `Graph` is made up of individual `Node`s which represent a single operation.
    It has a `target` which the operation this `Node` will execute.
    It has `args` and `kwargs` to execute its `target` with. These may contain other `Node`s and are therefore `dependencies` of this `Node`.
    Conversely this `Node` is a `listener` of its `dependencies`.
    
    During execution of the computation graph and therefore the `Node`s, each 
    
    Attributes:
        index (Optional[int]): Integer index of this `Node` within its greater computation graph.
        graph (Graph): 
        target (Union[Callable, Protocol]): Callable to execute as this `Node`'s operation. Might be a `Protocol` which is handled differently in node execution.
    """

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

        self.meta_data = self._meta_data()

        # If theres an alive Graph, add it.
        if self.attached:

            self.graph.add(self)
            
            # Preprocess args.
            self.preprocess()

    def __getstate__(self):

        state = self.__dict__.copy()

        return state
    
    def __setstate__(self, state: Dict) -> None:

        self.__dict__.update(state)
            
    @property
    def listeners(self) -> List[Self]:
        """Iterator from index to `Node`.

        Returns:
            List[Self]: List of listener `Node`s.
        """
        
        return [self.graph.nodes[index] for index in self._listeners]
    
    @property
    def dependencies(self) -> List[Self]:
        """Iterator from index to `Node`.

        Returns:
            List[Self]: List of dependency `Node`s.
        """
        
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
        """Checks to see if the `Graph` this `Node` is a part of is alive..
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
                args, kwargs = self.prepare_inputs((self.args, self.kwargs))

                # Call the target to get value.
                output = self.target(*args, **kwargs)
    

                # Set value.
                self.set_value(output)
        except NNsightError as e:
            raise e
        except Exception as e:
            traceback_content = traceback.format_exc()
            raise NNsightError(str(e), self.index, traceback_content)

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
        """Returns a Set of indexes starting from this node, and recursively iterating over all the Node's listeners.

        Args:
            subgraph (Optional[Set[int]], optional): Current subgraph. Defaults to None.

        Returns:
            Set[int]: Set of Node indexes.
        """

        if subgraph is None:
            subgraph = set()

        if self.index in subgraph:
            return subgraph
    
        subgraph.add(self.index)

        for listener in self.listeners:
            listener.subgraph(subgraph)

        return subgraph
    
    def _meta_data(self) -> Dict[str, Any]:
        """ Creates a dictionary of meta-data for this node.
        Contains the following key-value pairs:
            - traceback: Optional[str]: If the Graph is in debug mode, 
                a traceback string is compiled to be used if the execution of this Node raises an error.
        
        Returns:
            Dict[str, Any]: Meta-Data dictionary.
        """

        meta_data = dict()

        def traceback_str() -> str:
            """ Compiles a string of all the lines in the Traceback up until nnsight code is called.
            Returns:
                Str: Call Stack
            """
            traceback_str = ""
            stack = traceback.extract_stack()
            for frame in stack:
                # exclude frames created by nnsight or from the python environment
                if not bool(re.search((r'/lib/python3\.\d+/'), frame.filename)) and not ('/nnsight/src/nnsight/' in frame.filename):
                    traceback_str += f"  File \"{frame.filename}\", line {frame.lineno}, in {frame.name}\n"
                    traceback_str += f"    {frame.line}\n"
                else:
                    if traceback_str == "":
                        continue
                    else:
                        break

            traceback_str = "Traceback (most recent call last):\n" + traceback_str

            return traceback_str

        if self.attached and self.graph.debug:
            meta_data["traceback"] = traceback_str()

        return meta_data

    ### Magic Methods #####################################
    def __str__(self) -> str:
        return f"{self.target.__name__} {self.index}"

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__} at {hex(id(self))}&gt;"

    def __hash__(self) -> int:
        return id(self)


NodeType = TypeVar("NodeType", bound=Node)

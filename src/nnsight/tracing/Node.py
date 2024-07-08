from __future__ import annotations

import inspect
import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from .. import util
from ..logger import logger
from . import protocols
from .Proxy import Proxy

if TYPE_CHECKING:
    from .Graph import Graph
    from graphviz import Digraph


class Node:
    """A Node represents some action that should be carried out during execution of a Graph.

    Attributes:
        name (str): Unique name of node.
        graph (Graph): Weak reference to parent Graph object.
        proxy (Proxy): Weak reference to Proxy created from this Node.
        proxy_value (Any): Fake Tensor version of value. Used when graph has validate = True to check of Node actions are possible.
        target (Union[Callable, str]): Function to execute or name of Protocol.
        args (List[Any], optional): Positional arguments. Defaults to None.
        kwargs (Dict[str, Any], optional): Keyword arguments. Defaults to None.
        listeners (List[Node]): Nodes that depend on this node.
        dependencies (List[Node]): Nodes that this node depends on.
        value (Any): Actual value to be populated during execution.
    """

    def __init__(
        self,
        target: Union[Callable, str],
        graph: "Graph" = None,
        proxy_value: Any = inspect._empty,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        name: str = None,
    ) -> None:
        super().__init__()

        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()

        args = list(args)

        self.graph: "Graph" = graph
        self.proxy_value = proxy_value
        self.target = target
        self.args, self.kwargs = args, kwargs

        self.proxy: Optional[Proxy] = None

        self._value: Any = inspect._empty

        self.listeners: List[Node] = list()
        self.dependencies: List[Node] = list()

        self.remaining_listeners = 0
        self.remaining_dependencies = 0

        # Preprocess args.
        self.preprocess()

        # Node.graph is a weak reference to avoid reference loops.
        self.graph = weakref.proxy(self.graph) if self.graph is not None else None

        self.name: str = name

        # If theres an alive Graph, add it.
        if self.attached():

            self.graph.add(self)

    def preprocess(self) -> None:
        """Preprocess Node.args and Node.kwargs."""

        if self.attached() and protocols.BridgeProtocol.has_bridge(self.graph):

            bridge = protocols.BridgeProtocol.get_bridge(self.graph)

            # Protocol nodes don't redirect execution to the current context's graph by default
            redirect_execution = (
                self.target.redirect
                if isinstance(self.target, type)
                and issubclass(self.target, protocols.Protocol)
                else True
            )
            if redirect_execution:
                self.graph = bridge.peek_graph()

        def preprocess_node(node: Union[Node, Proxy]):

            if isinstance(node, Proxy):

                node = node.node

            if self.attached() and self.graph.id != node.graph.id:

                node = protocols.BridgeProtocol.add(node, self.graph).node

            if not node.done():

                self.dependencies.append(node)
                # Weakref so no reference loop
                node.listeners.append(weakref.proxy(self))

            # Otherwise just get the value if its already done.
            else:

                node = node.value

            return node

        self.args, self.kwargs = util.apply(
            (self.args, self.kwargs), preprocess_node, (Node, Proxy)
        )

    @property
    def value(self) -> Any:
        """Property to return the value of this node.

        Returns:
            Any: The stored value of the node, populated during execution of the model.

        Raises:
            ValueError: If the underlying ._value is inspect._empty (therefore never set or destroyed).
        """

        if not self.done():
            raise ValueError("Accessing value before it's been set.")

        return self._value

    def attached(self) -> bool:
        """Checks to see if the weakref to the Graph is alive or dead.

        Returns:
            bool: Is Node attached.
        """

        try:

            return self.graph.alive

        except:
            return False

    def create(
        self,
        target: Union[Callable, str],
        proxy_value: Any = inspect._empty,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        name: str = None,
    ) -> Union[Proxy, Any]:
        """We use Node.add vs Graph.add in case graph is dead.
        If the graph is dead, we assume this node is ready to execute and therefore we try and execute it and then return its value.

        Returns:
            Union[Proxy, Any]: Proxy or value
        """

        if not self.attached():

            # Create Node with no values or Graph.
            node = Node(
                target=target,
                graph=None,
                proxy_value=None,
                args=args,
                kwargs=kwargs,
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
            target=target,
            name=name,
            proxy_value=proxy_value,
            args=args,
            kwargs=kwargs,
        )

    def reset(self) -> None:
        """Resets this Nodes remaining_listeners and remaining_dependencies."""

        self.remaining_listeners = len(self.listeners)
        self.remaining_dependencies = len(self.dependencies)

    def done(self) -> bool:
        """Returns true if the value of this node has been set.

        Returns:
            bool: If done.
        """
        return self._value is not inspect._empty

    def fulfilled(self) -> bool:
        """Returns true if remaining_dependencies is 0.

        Returns:
            bool: If fulfilled.
        """
        return self.remaining_dependencies == 0

    def redundant(self) -> bool:
        """Returns true if remaining_listeners is 0.

        Returns:
            bool: If redundant.
        """
        return self.remaining_listeners == 0

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
                return node.proxy_value

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
            
            raise Exception(f"Above exception when execution Node: '{self.name}' in Graph: '{self.graph.id}'") from e

    def set_value(self, value: Any) -> None:
        """Sets the value of this Node and logs the event.
        Updates remaining_dependencies of listeners. If they are now fulfilled, execute them.
        Updates remaining_listeners of dependencies. If they are now redundant, destroy them.

        Args:
            value (Any): Value.
        """
        self._value = value

        logger.info(f"=> SET({self.name})")

        for listener in self.listeners:
            listener.remaining_dependencies -= 1

            if listener.fulfilled():
                listener.execute()

        for dependency in self.dependencies:
            dependency.remaining_listeners -= 1

            if dependency.redundant():
                dependency.destroy()

        if self.done() and self.redundant():
            self.destroy()

    def destroy(self) -> None:
        """Removes the reference to the node's value and logs it's destruction."""

        logger.info(f"=> DEL({self.name})")

        self._value = inspect._empty

    def visualize(
            self, 
            graph_viz: "Digraph", 
            arg_value_count: int, 
            is_arg: bool
            ) -> Tuple[str, int]:
        """ Visualizes this node by adding it to Digraph visual object. Can handle adding dependency arguments as well as edges between them.

        Args:
            graph_viz (Digraph): Visualization graph object.
            arg_value_count (int): Total count of non-Node arguments in the Digraph so far.
            is_arg (bool): If True, node will add itself to the graph without looping over its dependencies. 
                           Nodes are responsible for visualizing their arguments only when they are called to visualize directly from the main loop in Graph.viz()
        
        Returns:
            str: Node name.
            int: Count of value argument added to the Digraph so far.
        """
        
        # Visualization of protocol nodes is delegated to their respective classes
        if isinstance(self.target, type) and issubclass(self.target, protocols.Protocol):
            arg_value_count = self.target.visualize(self, 
                                                    graph_viz, 
                                                    arg_value_count=arg_value_count, 
                                                    is_arg=is_arg)
        else:
            # Adding current node
            graph_viz.node(self.name, 
                           label=self.target.__name__, 
                           **{"color": "black", "shape": "ellipse"})

            if not is_arg:
                base_arg_style = {"color": "gray", "shape": "box"}
                for i, arg in enumerate(self.args):
                    name, arg_value_count = util.add_arg_to_viz(graph_viz, 
                                                                arg, 
                                                                Node,
                                                                arg_value_count, 
                                                                base_arg_style)
                    
                    graph_viz.edge(name, self.name)

                for key, arg in self.kwargs.items():
                    name, arg_value_count = util.add_arg_to_viz(graph_viz, 
                                                                arg, 
                                                                Node,
                                                                arg_value_count, 
                                                                base_arg_style, 
                                                                key)
                    graph_viz.edge(name, self.name)
        
        return self.name, arg_value_count

    def __str__(self) -> str:
        args = util.apply(self.args, lambda x: f"'{x}'", str)
        args = util.apply(args, lambda x: x.name, Node)
        args = [str(arg) for arg in args]
        return f"{self.name}:[args:({','.join(args)}) l:{len(self.listeners)} d:{len(self.dependencies)}]"

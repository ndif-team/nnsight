from __future__ import annotations

import inspect
import weakref
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch

from .. import util
from ..logger import logger
from . import protocols
from .Proxy import Proxy

if TYPE_CHECKING:
    from .Graph import Graph

    try:
        from pygraphviz import AGraph
    except:
        pass


class Node:
    """A Node represents some action that should be carried out during execution of a Graph.

    The class represents the operations (and the resulting output of said operations) they are tracing AND nodes that actually execute the operations when executing the Graph. The Nodes you are Tracing are the same object as the ones that are executed.

        * Nodes have a ``.proxy_value`` attribute that are a result of the tracing operation, and are FakeTensors allowing you to view the shape and datatypes of the actual resulting value that will be populated when the node' operation is executed.
        * Nodes carry out their operation in ``.execute()`` where their arguments are pre-processed and their value is set in ``.set_value()``.
        * Arguments passed to the node are other nodes, where a bi-directional dependency graph is formed. During execution pre-processing, the arguments that are nodes and converted to their value.
        * Nodes are responsible for updating their listeners that one of their dependencies are completed, and if all are completed that they should execute. Similarly, nodes must inform their dependencies when one of their listeners has ceased "listening." If the node has no listeners, it's value is destroyed by calling ``.destroy()`` in order to free memory. When re-executing the same graph and therefore the same nodes, the remaining listeners and dependencies are reset on each node.

    Attributes:
        name (str): Unique name of node.
        graph (Graph): Weak reference to parent Graph object.
        proxy (Proxy): Weak reference to Proxy created from this Node.
        proxy_value (Any): Fake Tensor version of value. Used when graph has validate = True to check of Node actions are possible.
        target (Union[Callable, str]): Function to execute or name of Protocol.
        args (List[Any], optional): Positional arguments. Defaults to None.
        kwargs (Dict[str, Any], optional): Keyword arguments. Defaults to None.
        listeners (List[Node]): Nodes that depend on this node.
        arg_dependencies (List[Node]): Nodes that this node depends on.
        cond_dependency (Optional[Node]): ConditionalProtocol node if this node was defined within a Conditional context.
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
        self.arg_dependencies: List[Node] = list()
        self.cond_dependency: Optional[Node] = None

        self.remaining_listeners = 0
        self.remaining_dependencies = 0

        # Preprocess args.
        self.preprocess()

        # Node.graph is a weak reference to avoid reference loops.
        self.graph = (
            weakref.proxy(self.graph) if self.graph is not None else None
        )

        self.name: str = name

        # If theres an alive Graph, add it.
        if self.attached():

            self.graph.add(self)

    def preprocess(self) -> None:
        """Preprocess Node.args and Node.kwargs."""

        # bridge graph redirection
        if self.attached():
            self.graph = (
                protocols.BridgeProtocol.peek_graph(self.graph)
                if (
                    self.target.redirect
                    if isinstance(self.target, type)
                    and issubclass(self.target, protocols.Protocol)
                    else True
                )
                else self.graph
            )

        def preprocess_node(node: Union[Node, Proxy]):

            if isinstance(node, Proxy):

                node = node.node

            if node.done():

                return node.value

            if self.attached() and self.graph.id != node.graph.id:

                node = protocols.BridgeProtocol.add(node).node

            self.arg_dependencies.append(node)
            # Weakref so no reference loop
            node.listeners.append(weakref.proxy(self))

            return node

        self.args, self.kwargs = util.apply(
            (self.args, self.kwargs), preprocess_node, (Node, Proxy)
        )

        # conditional context handling
        if (
            self.attached()
            and protocols.ConditionalProtocol.has_conditional(self.graph)
            and (
                self.target.condition
                if isinstance(self.target, type)
                and issubclass(self.target, protocols.Protocol)
                else True
            )
        ):

            conditional_node = protocols.ConditionalProtocol.peek_conditional(
                self.graph
            )

            # only the top dependency needs to add the Conditional as a dependency
            # if none of the dependent are dependent on the Conditional, then add it
            if conditional_node:
                if all(
                    [
                        not protocols.ConditionalProtocol.is_node_conditioned(
                            arg
                        )
                        for arg in self.arg_dependencies
                    ]
                ):
                    self.cond_dependency = conditional_node
                    conditional_node.listeners.append(weakref.proxy(self))

                protocols.ConditionalProtocol.add_conditioned_node(self)

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

            graph: "Graph" = None

            def find_attached_graph(node: Union[Proxy, Node]):

                if isinstance(node, Proxy):

                    node = node.node

                nonlocal graph

                if node.attached():

                    graph = node.graph

            util.apply((args, kwargs), find_attached_graph, (Proxy, Node))

            if graph is not None:

                return graph.create(
                    target=target,
                    name=name,
                    proxy_value=proxy_value,
                    args=args,
                    kwargs=kwargs,
                )

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
        self.remaining_dependencies = len(self.arg_dependencies) + int(
            not (self.cond_dependency is None)
        )

    def done(self) -> bool:
        """Returns true if the value of this node has been set.

        Returns:
            bool: If done.
        """
        return self._value is not inspect._empty

    def executed(self) -> bool:
        """Returns true if remaining_dependencies is less than 0.

        Returns:
            bool: If executed.
        """
        return self.remaining_dependencies < 0

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

            raise type(e)(
                f"Above exception when execution Node: '{self.name}' in Graph: '{self.graph.id}'"
            ) from e

        finally:
            self.remaining_dependencies -= 1

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

            if listener.fulfilled() and not self.graph.sequential:
                listener.execute()

        for dependency in self.arg_dependencies:
            dependency.remaining_listeners -= 1

            if dependency.redundant():
                dependency.destroy()

        if self.done() and self.redundant():
            self.destroy()

    def destroy(self) -> None:
        """Removes the reference to the node's value and logs it's destruction."""

        logger.info(f"=> DEL({self.name})")

        self._value = inspect._empty

    def clean(self) -> None:
        """Clean up dependencies during early execution stop"""

        # BridgeProtocol nodes must clean up their corresponding external proxy
        if isinstance(self.target, type) and issubclass(
            self.target, protocols.BridgeProtocol
        ):
            bridge = protocols.BridgeProtocol.get_bridge(self.graph)
            lock_node = bridge.get_graph(self.args[0]).nodes[self.args[1]]
            lock_dependency = lock_node.args[0]
            lock_dependency.remaining_listeners -= 1
            lock_node.destroy()
            if lock_dependency.redundant():
                lock_dependency.destroy()
        else:
            for dependency in self.arg_dependencies:
                dependency.remaining_listeners -= 1
                if dependency.redundant():
                    dependency.destroy()

    def visualize(
        self, viz_graph: "AGraph", recursive: bool, backend_name: str = ""
    ) -> str:
        """Adds this node to the visualization graph and recursively visualizes its arguments and adds edges between them.

        Args:
            - viz_graph (AGraph): Visualization graph.
            - recursive (bool): If True, recursively visualizes all sub-graphs.
            - backend_name (str): Inherent parent graph name for unique differentiation in recursive visualization.

        Returns:
            - str: name of this node.
        """

        styles = {
            "node": {"color": "black", "shape": "ellipse"},
            "label": (self.target if isinstance(self.target, str) else self.target.__name__),
            "arg": defaultdict(lambda: {"color": "gray", "shape": "box"}),
            "arg_kname": defaultdict(lambda: None),
            "edge": defaultdict(lambda: "solid"),
        }

        node_name = backend_name + self.name

        if isinstance(self.target, type) and issubclass(
            self.target, protocols.Protocol
        ):
            styles = self.target.style()
            viz_graph.add_node(node_name, label=styles["label"], **styles["node"])
            if (
                recursive
                and self.target == protocols.LocalBackendExecuteProtocol
            ):
                # recursively draw all sub-graphs
                for sub_node in self.args[0].graph.nodes.values():
                    # draw root nodes and attach them to their LocalBackendExecuteProtocol node
                    if (
                        len(sub_node.arg_dependencies)
                        + int(not (sub_node.cond_dependency is None))
                    ) == 0:
                        sub_node_name = sub_node.visualize(
                            viz_graph, recursive, node_name + "_"
                        )
                        viz_graph.add_edge(
                            node_name,
                            sub_node_name,
                            style="dotted",
                            color="purple",
                        )
                    # draw bottom up
                    elif len(sub_node.listeners) == 0:
                        sub_node_name = sub_node.visualize(
                            viz_graph, recursive, node_name + "_"
                        )
        else:
            viz_graph.add_node(node_name, label=styles["label"], **styles["node"])

        def visualize_args(arg_collection):
            """Recursively visualizes the arguments of this node.

            Args:
                - arg_collection (Union[List[Any], Dict[str, Any]]): Collection of Node arguments.
            """

            for key, arg in arg_collection:
                if isinstance(arg, Node):
                    name = arg.visualize(viz_graph, recursive, backend_name)
                else:
                    # show link between iterable values with Node dependencies
                    iter_val_dependencies = []
                    if isinstance(arg, Iterable):
                        for element in arg:
                            if isinstance(element, Node):
                                dep_name = element.visualize(viz_graph, recursive, backend_name)
                                iter_val_dependencies.append(dep_name)
                    
                    name = node_name
                    if isinstance(arg, torch.Tensor):
                        name += f"_Tensor_{key}"
                        label = "Tensor"
                    elif isinstance(arg, str):
                        name += f"_{arg}_{key}"
                        label = f'"{arg}"'
                    else:
                        name += f"_{arg}_{key}"
                        label = str(arg)

                    if isinstance(key, int):
                        if not styles["arg_kname"][key] is None:
                            label = f"{styles['arg_kname'][key]}={label}"
                    else:
                        label = f"{key}={label}"

                    viz_graph.add_node(name, label=label, **styles["arg"][key])

                    for dep_name in iter_val_dependencies:
                        viz_graph.add_edge(dep_name, name, style="dashed", color="gray")

                viz_graph.add_edge(name, node_name, style=styles["edge"][key])

        visualize_args(enumerate(self.args))

        visualize_args(self.kwargs.items())

        if isinstance(self.cond_dependency, Node):
            name = self.cond_dependency.visualize(
                viz_graph, recursive, backend_name
            )
            viz_graph.add_edge(
                name, node_name, style=styles["edge"][None], color="#FF8C00"
            )

        return node_name

    def __str__(self) -> str:
        args = util.apply(self.args, lambda x: f"'{x}'", str)
        args = util.apply(args, lambda x: x.name, Node)
        args = [str(arg) for arg in args]

        return f"{self.name}:[args:({','.join(args)}) l:{len(self.listeners)} a_d:{len(self.arg_dependencies)} c_d{bool(self.cond_dependency)}]"

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__} at {hex(id(self))}&gt;"

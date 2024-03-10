from __future__ import annotations

import inspect
import weakref
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .. import util
from ..logger import logger
from . import protocols
from .protocols import PROTOCOLS
from .Proxy import Proxy

if TYPE_CHECKING:
    from .Graph import Graph


class Node:
    """A Node represents some action that should be carried out during execution of a Graph.

    Attributes:
        name (str): Unique name of node.
        graph (Graph): Reference to parent Graph object.
        proxy_value (Any): Meta version of value. Used when graph has validate = True.
        target (Union[Callable, str]): Function to execute or reserved string name.
        args (List[Any], optional): Positional arguments. Defaults to None.
        kwargs (Dict[str, Any], optional): Keyword arguments. Defaults to None.
        listeners (List[Node]): Nodes that depend on this node.
        dependencies (List[Node]): Nodes that this node depends on.
        value (Any): Actual value to be populated during execution.
    """

    @staticmethod
    def prepare_proxy_values(values, device: torch.device = None):
        """Prepare arguments for validating a node's target.
        Converts Proxies and Nodes to their proxy_value and moves tensors to 'meta' device.

        Args:
            values (Any): Values to prepare.
            device (torch.device): Device to try and move all tensors to. If None, moves all tensors to device of first tensor.
        Returns:
            values (Any): Prepared values.
        """

        def slice_to_value(arg: slice):
            return slice(
                Node.prepare_proxy_values(arg.start),
                Node.prepare_proxy_values(arg.stop),
                Node.prepare_proxy_values(arg.step),
            )

        # Convert proxies to their proxy_value
        values = util.apply(values, lambda x: x.node.proxy_value, Proxy)
        # Convert nodes to their proxy_value
        values = util.apply(values, lambda x: x.proxy_value, Node)
        # Slices may have proxies as part of their attributes so convert those to their proxy_values
        values = util.apply(values, slice_to_value, slice)

        if device is None:

            # Arguments might be tensors created outside of scanning. Also the model might be a 'meta' pre-dispatched version of the model.
            # That means the tensors as args and the model are different devices but we dont want to have to have the users move tensors to 'meta'
            # So only when theres a FakeTensor with device meta, we move other tensors also to meta.

            def get_device(tensor: torch.Tensor):

                nonlocal device

                if (
                    device is None
                    and isinstance(tensor, FakeTensor)
                    and tensor.device.type == "meta"
                ):

                    device = tensor.device.type

            util.apply(values, get_device, torch.Tensor)

        if device is not None:

            values = util.apply(values, lambda x: x.to(device), torch.Tensor)

        return values

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

        graph = weakref.proxy(graph) if graph is not None else None

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

        self.preprocess()

        self.name: str = name

        if self.attached():

            self.graph.add(self)

    def preprocess(self) -> None:

        max_rank = None
        max_graph = self.graph

        if self.attached() and protocols.BridgeProtocol.has_bridge(self.graph):

            bridge = protocols.BridgeProtocol.get_bridge(self.graph)
            max_rank = bridge.rank(self.graph)

        else:

            bridge = None

        def preprocess_arg(arg: Any):

            nonlocal bridge
            nonlocal max_rank
            nonlocal max_graph

            if isinstance(arg, Proxy):

                arg = arg.node

            if isinstance(arg, Node):

                if bridge is not None:

                    graph = arg.graph
                    rank = bridge.rank(graph)

                    if rank > max_rank:

                        max_rank = rank
                        max_graph = graph

            return arg

        self.args, self.kwargs = util.apply(
            (self.args, self.kwargs), preprocess_arg, (Proxy, Node)
        )

        self.graph = max_graph

        def preprocess_arg2(node: Node):

            if self.graph is not node.graph:

                if self.attached() and protocols.BridgeProtocol.has_bridge(node.graph):

                    node = protocols.BridgeProtocol.add(node, self).node

                else:
                    # TODO error
                    pass
                
            if not node.done():

                self.dependencies.append(node)
                node.listeners.append(self)

            return node

        self.args, self.kwargs = util.apply(
            (self.args, self.kwargs), preprocess_arg2, Node
        )

    @property
    def value(self) -> Any:
        """Property to return the value of this node.

        Returns:
            Any: The stored value of the node, populated during execution of the model.
        """

        if not self.done():
            raise ValueError("Accessing value before it's been set.")

        return self._value

    def attached(self) -> bool:
        """Checks to see if the weakref to the Graph is aliveor dead.

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
        proxy_value: Any = None,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        name: str = None,
    ) -> Union[Proxy, Any]:
        """We use Node.add vs Graph.add in case graph is dead.
        If the graph is dead, we assume this node is ready to execute and therfore we try and execute it and then return its value.

        Returns:
            Union[Proxy, Any]: Proxy or value
        """

        if not self.attached():

            node = Node(
                target=target,
                graph=None,
                proxy_value=None,
                args=args,
                kwargs=kwargs,
            )
            
            node.reset()

            # So it doesn't get destroyed.
            node.remaining_listeners = 1
            
            node.compile()

            value = node.value

            node.destroy()

            return value

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

    def compile(self) -> None:
        """If fulfilled, execute the node, otherwise set its value to empty."""

        if self.fulfilled():
            
            self.execute()

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

    def prepare_inputs(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Prepare arguments for executing this node's target.
        Converts Nodes in args and kwargs to their value and moves tensors to correct device.


        Returns:
            Tuple[List[Any], Dict[str, Any]]: Prepared args and kwargs
        """

        # Turn nodes into their value
        def _value(node: Node):
            return node.value

        args, kwargs = util.apply((self.args, self.kwargs), _value, Node)

        device = None

        def _device(value: torch.Tensor):
            nonlocal device

            if device is None:
                device = value.device

        util.apply((args, kwargs), _device, torch.Tensor)

        def _to(value: torch.Tensor):
            return value.to(device)

        util.apply((args, kwargs), _to, torch.Tensor)

        return args, kwargs

    def execute(self) -> None:
        """Actually executes this node.
        Lets protocol execute if target is str.
        Else prepares args and kwargs and passed them to target.
        """

        if isinstance(self.target, str):

            # TODO error if not in protocols?

            PROTOCOLS[self.target].execute(self)

        else:

            # Prepare arguments.
            args, kwargs = self.prepare_inputs()

            # Call the target to get value.
            output = self.target(*args, **kwargs)

            self.set_value(output)

    def set_value(self, value: Any):
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

    def __str__(self) -> str:
        args = util.apply(self.args, lambda x: f"'{x}'", str)
        args = util.apply(args, lambda x: x.name, Node)
        args = [str(arg) for arg in args]
        return f"{self.name}:[args:({','.join(args)}) l:{len(self.listeners)} d:{len(self.dependencies)}]"

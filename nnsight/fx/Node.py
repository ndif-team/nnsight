from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import torch.futures

from .. import util
from ..logger import logger
from .Proxy import Proxy

if TYPE_CHECKING:
    from .Graph import Graph


class Node:
    """_summary_

    Attributes:
        name (str): _description_
        graph (Graph): _description_
        proxy_value (Any): _description_
        target (Union[Callable, str]): _description_
        args (List[Any], optional): _description_. Defaults to None.
        kwargs (Dict[str, Any], optional): _description_. Defaults to None.
        meta (Dict[str, Any], optional): _description_. Defaults to None.
        listeners (List[Node]): desc
        dependencies (List[Node]): desc
        _future (torch.futures.Future): desc
        _proxy_device (torch.device): desc
    """

    @staticmethod
    def update(value1, value2) -> None:
        """Updates Tensor values with other Tensor values.

        Args:
            value1 (_type_): _description_
            value2 (_type_): _description_
        """
        if isinstance(value1, torch.Tensor):
            value1[:] = value2
        elif isinstance(value1, list) or isinstance(value1, tuple):
            for value_idx in range(len(value1)):
                Node.update(value1[value_idx], value2[value_idx])
        elif isinstance(value1, dict):
            for key in value1:
                Node.update(value1[key], value2[key])

    @staticmethod
    def target_name(target) -> str:
        if isinstance(target, str):
            name = target
        elif callable(target):
            name = target.__name__

        return name

    def __init__(
        self,
        name: str,
        graph: "Graph",
        value: Any,
        target: Union[Callable, str],
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        meta: Dict[str, Any] = None,
    ) -> None:
        super().__init__()

        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        if meta is None:
            meta = dict()

        self.name = name
        self.graph = graph
        self.proxy_value = value
        self.target = target
        self.args = util.apply(args, lambda x: x.node, Proxy)
        self.kwargs = util.apply(kwargs, lambda x: x.node, Proxy)
        self.meta = meta

        self.listeners: List[Node] = list([self])
        self.dependencies: List[Node] = list()

        # Add all arguments that are nodes to nodes dependencies
        util.apply(self.args, lambda x: self.dependencies.append(x), Node)
        util.apply(self.kwargs, lambda x: self.dependencies.append(x), Node)
        # Add node to all arguments that are nodes' listeners
        util.apply(self.args, lambda x: x.listeners.append(self), Node)
        util.apply(self.kwargs, lambda x: x.listeners.append(self), Node)

        self._future: torch.futures.Future = None
        self._proxy_device: torch.device = None

    @property
    def future(self) -> torch.futures.Future:
        """Lazy creation of _future attribute.

        Returns:
            torch.futures.Future: _description_
        """
        if self._future is None:
            self._future = torch.futures.Future()

        return self._future

    @property
    def proxy_device(self) -> torch.device:
        """Lazy creation of _proxy_device attribute.

        Returns:
            torch.Device: _description_
        """
        if self._proxy_device is None:
            device = None

            def _device(value):
                nonlocal device
                device = value.device

            util.apply(self.proxy_value, _device, torch.Tensor)
            # TODO
            # util.apply(self.proxy_value, _device, torch.nn.Module)

            self._proxy_device = device

        return self._proxy_device

    def prepare_proxy_values(self, values):
        def slice_to_value(arg: slice):
            return slice(
                self.prepare_proxy_values(arg.start),
                self.prepare_proxy_values(arg.stop),
                self.prepare_proxy_values(arg.step),
            )

        # Convert procies to their proxy_value
        values = util.apply(values, lambda x: x.node.proxy_value, Proxy)
        # Slices may have proxies as part of their attributes so convert those to their proxy_values
        values = util.apply(values, slice_to_value, slice)
        # Move tensors to that of the proxy_device (probably 'meta')
        values = util.apply(values, lambda x: x.to(self.proxy_device), torch.Tensor)

        return values

    def compile(self) -> None:
        # When this future is done, log that event.
        self.future.add_done_callback(lambda x: logger.debug(f"=> SET({self.name})"))

        # Nodes tell listeners when to try and be executed.
        # This chains futures so after this node's future is done, it goes through
        # it's listeners in order and calls their .chain() method.
        future = self.listeners[0].future

        for listener in self.listeners[1:]:
            future = future.then(listener.chain)

        # Collect all listeners futures into a single future that when done, call this
        # nodes .destroy() method.
        torch.futures.collect_all(
            util.apply(self.listeners, lambda x: x.future, Node)
        ).add_done_callback(lambda x: self.destroy())

    def value(self) -> Any:
        """Wrapper for this node's future .value()

        Returns:
            Any: _description_
        """
        return self.future.value()

    def done(self) -> bool:
        """Wrapper for this node's future .done()

        Returns:
            bool: _description_
        """
        return self.future.done()

    def fufilled(self) -> bool:
        """Returns True if all of this node's dependencies are done.

        Returns:
            bool: _description_
        """
        for dependency in self.dependencies:
            if not dependency.done():
                return False

        return True

    def prepare_inputs(self) -> Tuple[List[Any], Dict[str, Any]]:
        # Turn futures into their value
        def _value(value: Node):
            return value.value()

        args = util.apply(self.args, _value, Node)
        kwargs = util.apply(self.kwargs, _value, Node)

        device = None

        def _device(value):
            nonlocal device
            device = value.device

        all_args = list(args) + list(kwargs.values())

        util.apply(list(reversed(all_args)), _device, torch.Tensor)
        # util.apply(list(reversed(all_args)), _device, torch.nn.Module)

        # Move tensors to device
        def _to(value: torch.Tensor):
            return value.to(device)

        args = util.apply(args, _to, torch.Tensor)
        kwargs = util.apply(kwargs, _to, torch.Tensor)

        return args, kwargs

    def execute(self) -> None:
        """Actually executes this node."""

        # We se a nodes target to 'null' if we don't want it to be executed and therefore never done
        if self.target == "null":
            return

        # Prepare arguments.
        args, kwargs = self.prepare_inputs()

        # If target is a string, it must be a method attribute on the first argument object.
        if isinstance(self.target, str):
            obj, *args = args

            target = getattr(obj, self.target)
        # Otherwise it must be the function itself.
        else:
            target = self.target

        # Call the target to get value.
        output = target(*args, **kwargs)

        # Set this nodes future value to result.
        self.future.set_result(output)

    def destroy(self) -> None:
        """Removes the reference to the node's _future and logs it's destruction."""
        logger.debug(f"=> DEL({self.name})")

        self._future = None

    def chain(self, future: torch.futures.Future):
        # If all of a nodes dependencies are done, execute it.
        # Dont execute if already done.
        if self.fufilled() and not self.done():
            try:
                self.execute()
            except Exception as e:
                # TODO
                # An exectption is actually never thrown upward to the point it stops the program. Need to find a way.
                logger.exception(f"Exception in execution of node '{self.name}'.")

                self.future.set_exception(e)
                future.set_exception(e)

                raise e

        future.set_result(None)

    def __str__(self) -> str:
        args = util.apply(self.args, lambda x: f"'{x}'", str)
        args = util.apply(args, lambda x: x.name, Node)
        args = [str(arg) for arg in args]
        meta = f"{self.meta['file']}({self.meta['line']})" if self.meta else ""
        return f"{self.name}:[ {meta} args:({','.join(args)}) l:{len(self.listeners)} d:{len(self.dependencies)}]"

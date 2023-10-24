from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Type, Union

import torch

from .. import util
from .Node import Node
from ..patching import Patcher, Patch
from .Proxy import Proxy, proxy_wrapper


class Graph:
    """Represents a computation graph involving a Module

    Attributes:
        proxy_class (Type[Proxy]): Proxy class to use. Defaults to Proxy.
        nodes (Dict[str, Node]): Mapping of node name to node.
        name_idx (Dict[str, int]): Mapping of node target_name to number of previous names with the same target_name.
            Used so names are unique.
        module_proxy (Proxy): Proxy for given root module
        argument_node_names (Dict[str, List[str]]): _description_
        generation_idx (int): _description_

    """

    @staticmethod
    def trace(
        module: torch.nn.Module, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> Graph:
        """Given a module and some default (should be meta tensors) arguments, create a graph from the module's
        forward method.

        Args:
            module (torch.nn.Module): _description_
            args (List[Any]): desc
            kwargs (Dict[str, Any]): desc

        Returns:
            Graph: _description_
        """

        # Create a graph with the module as the root module
        graph = Graph(module)

        # Get 'unbound' version of forward method so we can pass in proxy of module insead of self
        forward = module.__class__.forward

        # Want list not tuple
        args = list(args)

        # Inspect forward signature to collect all parameters
        signature = inspect.signature(forward)

        def get_argument_value(param: inspect.Parameter, idx: int):
            """Gets the correct argument to pass to forward method.


            Args:
                param (_type_): _description_
                idx (_type_): _description_

            Returns:
                _type_: _description_
            """

            # If idx in range of provided args, create a proxy for that arg instead of default.
            if idx < len(args):
                return graph.add(
                    graph=graph, value=args[idx], target="argument", args=[param.name]
                )
            # If param name in provided kwargs, create a proxy for that arg instead of default.
            if param.name in kwargs:
                return graph.add(
                    graph=graph,
                    value=kwargs[param.name],
                    target="argument",
                    args=[param.name],
                )
            # Otherwise just return default
            return param.default

        # Create the appropriate proxies/values for the forward method in order to trace.
        arguments = [
            get_argument_value(param, i)
            for i, param in enumerate(list(signature.parameters.values())[1:])
        ]

        # Some methods cannot be caught because they arent torch functions or dont play nice with __torch_function__.
        # So the patcher repalces the methods with something to catch proxies and return proxies.
        with Patcher() as patcher:
            patcher.add(Patch(torch.full, proxy_wrapper(torch.full)))
            patcher.add(Patch(torch.finfo, proxy_wrapper(torch.finfo)))
            patcher.add(Patch(torch.arange, proxy_wrapper(torch.arange)))

            # Run forward with root module proxy and arguments
            output: Proxy = forward(graph.module_proxy, *arguments)

            # Get proxy_value for return
            value = util.apply(output, lambda x: x.node.proxy_value, Proxy)

            # Create the 'rtn_0' return proxy
            return_proxy = graph.add(
                graph=graph, value=value, target=Graph.rtn, args=output
            )

            # This is how we tell the graph not to destroy a proxy after it's listeners are completed.
            # Create a 'null' proxy. The return proxy listens to the 'null' proxy with args=[return_proxy.node] but 'null' will never be completed.
            graph.add(
                graph=graph,
                value=None,
                target="null",
                args=[return_proxy.node],
            )

        return graph

    @staticmethod
    def rtn(*args, **kwargs):
        """
        Function to just pass through data for returning data in a graph forward method.

        Returns:
            _type_: _description_
        """

        return args

    def __init__(
        self, module: torch.nn.Module, proxy_class: Type[Proxy] = Proxy
    ) -> None:
        """_summary_

        Args:
            module (torch.nn.Module): _description_
            proxy_class (Type[Proxy], optional): _description_.
        """
        self.proxy_class = proxy_class

        self.nodes: Dict[str, Node] = dict()
        self.name_idx: Dict[str, int] = dict()

        self.module_proxy = self.add(graph=self, value=module, target="module")
        self.argument_node_names: Dict[str, List[str]] = dict()

        self.generation_idx = 0

    def increment(self) -> None:
        """Increments the generation_idx by one. Should be called by a forward hook on the model being used for generation."""
        self.generation_idx += 1

    def compile(self, module: torch.nn.Module) -> None:
        """Re-compile graph to prepare for a new execution of the graph.

        Args:
            module (torch.nn.Module): Module to be considered the root module of the graph.
        """

        # Remove nodes that have no effect.
        self.eliminate_dead_code()

        # Reset all node futures.
        for node in self.nodes.values():
            node._future = None
        # Compile nodes individually.
        for node in self.nodes.values():
            node.compile()

        self.generation_idx = 0

        # Setting the root module future kicks off the graph execution.
        self.nodes["module_0"].future.set_result(module)

    def add(
        self,
        graph: Graph,
        value: Any,
        target: Union[Callable, str],
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        name: str = None,
    ) -> Proxy:
        """Adds a node to the graph and returns it's proxy.

        Args:
            graph (Graph): _description_
            value (Any): 'meta' proxy value used for tracing the shapes and values.
            target (Union[Callable, str]): Either the function to call for this node, or a string that's the name of a method attribute on the first arg.
            args (List[Any], optional): _description_. Defaults to None.
            kwargs (Dict[str, Any], optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to None.

        Returns:
            Proxy: _description_
        """
        target_name = Node.target_name(target)

        if target_name not in self.name_idx:
            self.name_idx[target_name] = 0
        else:
            if target_name == "rtn":
                raise ValueError("Can only have one return ('rtn') node.")
            if target_name == "module":
                raise ValueError("Can only have one module node.")

        if name is None:
            name = f"{target_name}_{self.name_idx[target_name]}"

        self.name_idx[target_name] += 1

        stack = inspect.stack()
        proxy_frame = stack[2]

        node = Node(
            name=name,
            graph=graph,
            value=value,
            target=target,
            args=args,
            kwargs=kwargs,
            meta={"line": proxy_frame.lineno, "file": proxy_frame.filename},
        )

        self.nodes[name] = node

        if target_name == "argument":
            module_path = args[0]

            if module_path not in self.argument_node_names:
                self.argument_node_names[module_path] = []

            self.argument_node_names[module_path].append(name)

        return self.proxy(node)

    def proxy(self, node: Node) -> Proxy:
        """Returns proxy of node with specified proxy_class.

        Args:
            node (Node): _description_

        Returns:
            Proxy: _description_
        """
        return self.proxy_class(node)

    def eliminate_dead_code(self):
        # TODO
        pass

    def wrap(self, module: torch.nn.Module) -> torch.nn.Module:
        """Replaces the forward method of the given module with an execution of the module's graph.

        Args:
            module (torch.nn.Module): _description_

        Returns:
            torch.nn.Module: _description_
        """

        def forward(*args, **kwargs):
            # Compile the graph with the given module as the root module.
            self.compile(module)

            # Gets list of all argument nodes for this graph.
            argument_nodes_list = list(self.argument_node_names.values())

            # Sets the result of the argument nodes future for args.
            for i, arg in enumerate(args):
                self.nodes[argument_nodes_list[i][0]].future.set_result(arg)

            # And then for kwargs.
            for key in kwargs:
                if key in self.argument_node_names:
                    self.nodes[self.argument_node_names[key][0]].future.set_result(arg)

            # 'rtn_0' should have the value we need to return.
            return_value = self.nodes["rtn_0"].value()
            self.nodes["rtn_0"].destroy()
            return return_value

        # Repalce forward method with custom graph execution method.
        module.forward = forward

        return module

    def __str__(self) -> str:
        result = ""

        for name, node in self.nodes.items():
            result += f"  %{node}\n"

        return result

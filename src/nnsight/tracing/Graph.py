from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Type, Union
import torch
from .. import util
from ..patching import Patch, Patcher
from .Node import Node
from .Proxy import Proxy, proxy_wrapper


class Graph:
    """Represents a computation graph involving a torch.nn.module.

    Reserved target names:
    
    * 'module' : There should only be the single root module as a node in the graph for tracing. Added on __init__ and when compiling, the node's value is set to to be whatever module that is being interleaved with this computation graph.
    * 'argument' : There can be multiple argument nodes. Their first argument needs to be the argument name which acts as a key in graph.argument_node_names which maps to a list of names for nodes that depend on it's value. These nodes values need to be set outside of the computation graph as entry points to kick of the execution of the graph.
    * 'rtn' : Should only be one 'rtn' target named node as this is what is used.
    * 'null' : Null nodes never get executed and therefore their listeners never get destroyed.

    Attributes:
        validate (bool): If to execute nodes as they are added with their proxy values in order to check if the executions are possible (i.e shape errors etc). Defaults to True.
        proxy_class (Type[Proxy]): Proxy class to use. Defaults to Proxy.
        nodes (Dict[str, Node]): Mapping of node name to node.
        name_idx (Dict[str, int]): Mapping of node target_name to number of previous names with the same target_name.
            Used so names are unique.
        module_proxy (Proxy): Proxy for given root meta module.
        argument_node_names (Dict[str, List[str]]): Map of name of argument to name of nodes that depend on it.
        generation_idx (int): Current generation index.
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

        # Get 'unbound' version of forward method so we can pass in proxy of module instead of self
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

        # Some methods cannot be caught because they aren't torch functions or dont play nice with __torch_function__.
        # So the patcher replaces the methods with something to catch proxies and return proxies.
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
        self,
        module: torch.nn.Module,
        proxy_class: Type[Proxy] = Proxy,
        validate: bool = True,
    ) -> None:
        self.proxy_class = proxy_class
        self.validate = validate

        self.nodes: Dict[str, Node] = dict()
        self.name_idx: Dict[str, int] = dict()

        self.module_proxy = self.add(value=module, target="module")
        self.argument_node_names: Dict[str, List[str]] = dict()

        self.generation_idx = 0

    def increment(self) -> None:
        """Increments the generation_idx by one. Should be called by a forward hook on the model being used for generation."""
        self.generation_idx += 1

    def compile(self, module: torch.nn.Module) -> None:
        """Re-compile graph to prepare for a new execution of the graph.

        Compiles all nodes and sets generation_idx to 0.

        Finally, sets the "module_0" node's value to the module that is being interleaved.

        Args:
            module (torch.nn.Module): Module to be considered the root module of the graph.
        """

        # Remove nodes that have no effect.
        self.eliminate_dead_code()

        # Compile nodes individually.
        for node in self.nodes.values():
            node.compile()

        self.generation_idx = 0

        # Setting the root module kicks off the graph execution.
        self.nodes["module_0"].set_value(module)

    def add(
        self,
        target: Union[Callable, str],
        value: Any = inspect._empty,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        name: str = None,
    ) -> Proxy:
        """Adds a node to the graph and returns it's proxy.

        Args:
            value (Any): 'meta' proxy value used for tracing the shapes and values.
            target (Union[Callable, str]): Either the function to call for this node, or a string of a reserved target name.
            args (List[Any], optional): Positional arguments of node. Defaults to None.
            kwargs (Dict[str, Any], optional): Keyword arguments of node. Defaults to None.
            name (str, optional): Unique name of node. Otherwise pull name from target Defaults to None.

        Returns:
            Proxy: Proxy for the added node.

        Raises:
            ValueError: If more than one reserved "rtn" or "module" nodes are added to the graph.
        """

        # If we're validating and the user did not provide a value, execute the given target with meta proxy values to compute new proxy_value.
        if self.validate and value is inspect._empty:
            _args = args if args is not None else []
            _kwargs = kwargs if kwargs is not None else {}

            value = target(
                *Node.prepare_proxy_values(_args),
                **Node.prepare_proxy_values(_kwargs),
            )

        target_name = target if isinstance(target, str) else target.__name__

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
            graph=self,
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
            node (Node): Node.

        Returns:
            Proxy: Proxy.
        """
        return self.proxy_class(node)

    def eliminate_dead_code(self):
        # TODO
        pass

    def wrap(self, module: torch.nn.Module) -> torch.nn.Module:
        """Replaces the forward method of the given module with an execution of the module's graph.

        Args:
            module (torch.nn.Module): Module to replace the forward method of.

        Returns:
            torch.nn.Module: The module, post-replacement.
        """

        def forward(*args, **kwargs):
            # Compile the graph with the given module as the root module.
            self.compile(module)

            # Gets list of all argument nodes for this graph.
            argument_nodes_list = list(self.argument_node_names.values())

            # Sets the result of the argument nodes for args.
            for i, arg in enumerate(args):
                self.nodes[argument_nodes_list[i][0]].set_value(arg)

            # And then for kwargs.
            for key in kwargs:
                if key in self.argument_node_names:
                    self.nodes[self.argument_node_names[key][0]].set_value(arg)

            # 'rtn_0' should have the value we need to return.
            return_value = self.nodes["rtn_0"].value
            self.nodes["rtn_0"].destroy()
            return return_value

        # Replace forward method with custom graph execution method.
        module.forward = forward

        return module

    def __str__(self) -> str:
        result = ""

        for name, node in self.nodes.items():
            result += f"  %{node}\n"

        return result

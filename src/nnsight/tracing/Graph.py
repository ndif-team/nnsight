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
    * 'swp' : swp nodes indicate populating the graph's swap attribute. When executed, its value is not set. Logic involving the swap value should set its value after using it.
    * 'null' : Null nodes never get executed and therefore their listeners never get destroyed.
    * 'grad' : grad nodes indicates adding a `.register_hook()` to a tensor proxy

    Attributes:
        validate (bool): If to execute nodes as they are added with their proxy values in order to check if the executions are possible (i.e shape errors etc). Defaults to True.
        proxy_class (Type[Proxy]): Proxy class to use. Defaults to Proxy.
        nodes (Dict[str, Node]): Mapping of node name to node.
        name_idx (Dict[str, int]): Mapping of node target_name to number of previous names with the same target_name.
            Used so names are unique.
        module_proxy (Proxy): Proxy for given root meta module.
        argument_node_names (Dict[str, List[str]]): Map of name of argument to name of nodes that depend on it.
        generation_idx (int): Current generation index.
        swap (Node): Attribute to store swap values from 'swp' nodes.
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

        trace_args = []
        trace_kwargs = {}

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
                trace_args.append(graph.add(
                    value=args[idx], target="argument", args=[param.name]
                ))
            # If param name in provided kwargs, create a proxy for that arg instead of default.
            elif param.name in kwargs and type(kwargs[param.name]) != type(param.default):
                trace_kwargs[param.name] = graph.add(
                    value=kwargs[param.name],
                    target="argument",
                    args=[param.name],
                )
            else:
                # Otherwise just return default
                trace_kwargs[param.name] = param.default

        # Create the appropriate proxies/values for the forward method in order to trace.
        arguments = [
            get_argument_value(param, i)
            for i, param in enumerate(list(signature.parameters.values())[1:])
        ]

        # Some methods cannot be caught because they aren't torch functions or dont play nice with __torch_function__.
        # So the patcher replaces the methods with something to catch proxies and return proxies.
        with Patcher() as patcher:
            patcher.add(Patch(torch, proxy_wrapper(torch.full), "full"))
            patcher.add(Patch(torch, proxy_wrapper(torch.finfo), "finfo"))
            patcher.add(Patch(torch, proxy_wrapper(torch.arange), "arange"))

            # Run forward with root module proxy and arguments
            output = forward(graph.module_proxy, *trace_args, **trace_kwargs)

        return graph

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

        self.swap: Node = None

    def get_swap(self, value):
        if self.swap is not None:
            device = None

            def _device(value: torch.Tensor):
                nonlocal device

                device = value.device

            util.apply(value, _device, torch.Tensor)

            value = util.apply(self.swap.args[1], lambda x: x.value, Node)

            if device is not None:

                def _to(value: torch.Tensor):
                    return value.to(device)

                value = util.apply(value, _to, torch.Tensor)

            # Set value of 'swp' node so it destroys itself and listeners.
            self.swap.set_value(True)

            # Un-set swap.
            self.swap = None

        return value

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
            ValueError: If more than one reserved "module" nodes are added to the graph.
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

        # (for when you want to apply things to proxies after model execution?)
        if not node.done():

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

            # should have the value we need to return.
            return_value = self.swap
            self.swap.set_value(True)
            return return_value

        # Replace forward method with custom graph execution method.
        module.forward = forward

        return module

    def vis(self, filename: str = "graph", format: str = "png"):
        import graphviz

        def style(value: Any) -> Dict[str, Any]:
            style = {}

            if isinstance(value, Node):
                if value.target == "null":
                    style["color"] = "red"

                elif value.target == "argument":
                    style["color"] = "green"

                elif value.target == "module":
                    style["color"] = "green4"

                else:
                    style["color"] = "black"
            else:
                style["color"] = "grey"
                style["shape"] = "box"

            return style

        arg_name_idx = 0

        def add_node(value: Any, graph: graphviz.Digraph, kname: str = None) -> str:
            nonlocal arg_name_idx

            if isinstance(value, Node):
                name = value.name
                label = (
                    value.target
                    if isinstance(value.target, str)
                    else value.target.__name__
                )
            else:
                if isinstance(value, torch.Tensor):
                    name = str(arg_name_idx)
                    label = "Tensor"
                elif isinstance(value, str):
                    name = str(arg_name_idx)
                    label = f'"{value}"'
                else:
                    name = str(arg_name_idx)
                    label = str(value)

                arg_name_idx += 1

            if kname is not None:
                label = f"{kname}={label}"

            if f"\t{name}" not in graph.body:
                graph.node(name, label=label, **style(value))

            return name

        graph = graphviz.Digraph("round-table", comment="The Round Table")

        for node in self.nodes.values():
            add_node(node, graph)

            for i, arg in enumerate(node.args):
                kname = None

                if node.target == "argument":
                    if i == 0:
                        kname = "key"
                    elif i == 1:
                        kname = "batch_size"
                    elif i == 2:
                        kname = "batch_start"

                name = add_node(arg, graph, kname=kname)

                graph.edge(name, node.name)

            for kname, arg in node.kwargs.items():
                name = add_node(arg, graph, kname=kname)

                graph.edge(name, node.name)

        graph.render(filename=filename, format=format)

    def __str__(self) -> str:
        result = ""

        for name, node in self.nodes.items():
            result += f"  %{node}\n"

        return result

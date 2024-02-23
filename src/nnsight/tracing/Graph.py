from __future__ import annotations

import inspect
import weakref
from typing import Any, Callable, Dict, List, Type, Union

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .. import util
from .Node import Node
from .Proxy import Proxy


class Graph:
    """Represents a computation graph involving a torch.nn.module.

    Reserved target names:

    * 'argument' : There can be multiple argument nodes. Their first argument needs to be the argument name which acts as a key in graph.argument_node_names which maps to a list of names for nodes that depend on it's value. These nodes values need to be set outside of the computation graph as entry points to kick of the execution of the graph.
    * 'swap' : swp nodes indicate populating the graph's swap attribute. When executed, its value is not set. Logic involving the swap value should set its value after using it.
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
        swap (Node): Attribute to store swap values from 'swap' nodes.
    """

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

        self.argument_node_names: Dict[str, List[str]] = dict()

        self.swap: Node = None

        self.module_proxy = self.add(
            value=module, target="argument", args=["nnsight_root_module"]
        )

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

    def compile(self, module: torch.nn.Module) -> None:
        """Re-compile graph to prepare for a new execution of the graph.

        Compiles all nodes.

        Finally, sets the "nnsight_root_module" node's value to the module that is being interleaved.

        Args:
            module (torch.nn.Module): Module to be considered the root module of the graph.
        """

        # Remove nodes that have no effect.
        self.eliminate_dead_code()

        # Compile nodes individually.
        for node in self.nodes.values():
            node.compile()

        # Setting the root module kicks off the graph execution.
        self.module_proxy.node.set_value(module)

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

            with FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(assume_static_by_default=True),
            ) as fake_mode:

                try:

                    value = target(
                        *Node.prepare_proxy_values(_args),
                        **Node.prepare_proxy_values(_kwargs),
                    )

                except RuntimeError:

                    value = None

        target_name = target if isinstance(target, str) else target.__name__

        if target_name not in self.name_idx:
            self.name_idx[target_name] = 0

        if name is None:
            name = f"{target_name}_{self.name_idx[target_name]}"

        stack = inspect.stack()
        proxy_frame = stack[2]

        node = Node(
            name=name,
            graph=weakref.proxy(self),
            value=value,
            target=target,
            args=args,
            kwargs=kwargs,
            meta={"line": proxy_frame.lineno, "file": proxy_frame.filename},
        )

        self.name_idx[target_name] += 1

        self.nodes[name] = node

        if target_name == "argument":
            module_path = args[0]

            if module_path not in self.argument_node_names:
                self.argument_node_names[module_path] = []

            self.argument_node_names[module_path].append(name)

        return self.proxy_class(node)

    def eliminate_dead_code(self):
        # TODO
        pass

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

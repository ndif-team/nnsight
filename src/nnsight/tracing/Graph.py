from __future__ import annotations

import inspect
import weakref
from typing import Any, Callable, Dict, List, Type, Union

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

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
        tracing (bool): If currently tracing operations
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
        proxy_class: Type[Proxy] = Proxy,
        validate: bool = True,
        graph_id: int = None,
    ) -> None:

        self.id = graph_id or id(self)

        self.proxy_class = proxy_class
        self.validate = validate

        self.alive = True

        self.nodes: Dict[str, Node] = dict()
        self.name_idx: Dict[str, int] = dict()

        self.attachments = dict()

    def compile(self) -> None:
        """Re-compile graph to prepare for a new execution of the graph.

        Resets all nodes, then compiles all nodes.
        """

        # Reset nodes individually.
        for node in self.nodes.values():
            node.reset()

        # Compile nodes individually.
        for node in self.nodes.values():
            node.compile()

    def create(self, *args, **kwargs) -> Proxy:

        return self.proxy_class(Node(*args, graph=self, **kwargs))

    def add(self, node: Node) -> None:

        # If we're validating and the user did not provide a value, execute the given target with meta proxy values to compute new proxy_value.
        if self.validate and node.proxy_value is inspect._empty:
            _args = node.args if node.args is not None else []
            _kwargs = node.kwargs if node.kwargs is not None else {}

            with FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(assume_static_by_default=True),
            ) as fake_mode:
                with FakeCopyMode(fake_mode):

                    node.proxy_value = node.target(
                        *Node.prepare_proxy_values(_args),
                        **Node.prepare_proxy_values(_kwargs),
                    )

        name = node.target if isinstance(node.target, str) else node.target.__name__

        if name not in self.name_idx:
            self.name_idx[name] = 0

        if node.name is None:
            node.name = f"{name}_{self.name_idx[name]}"

        self.name_idx[name] += 1

        self.nodes[node.name] = node

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

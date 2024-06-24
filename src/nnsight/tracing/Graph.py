from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Type, Union

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .Node import Node
from .Proxy import Proxy
from .protocols import Protocol, LockProtocol
from ..util import apply


class Graph:
    """Represents a computation graph composed of Nodes


    Attributes:
        validate (bool): If to execute nodes as they are added with their proxy values in order to check if the executions are possible (i.e shape errors etc). Defaults to True.
        proxy_class (Type[Proxy]): Proxy class to use. Defaults to Proxy.
        alive (bool): If this Graph should be considered alive, and therefore added to. Used by Nodes.
        nodes (Dict[str, Node]): Mapping of node name to node.
        name_idx (Dict[str, int]): Mapping of node target_name to number of previous names with the same target_name.
            Used so names are unique.
        attachments (Dict[str, Any]): Dictionary object used to add extra functionality to this Graph. Used by Protocols.
    """

    def __init__(
        self,
        proxy_class: Type[Proxy] = Proxy,
        validate: bool = True,
        graph_id: int = None,
        default: bool = False,
    ) -> None:

        self.id = graph_id or id(self)

        self.proxy_class = proxy_class
        self.validate = validate

        self.alive = True
        self.default = default

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
        """Creates a Node directly on this Graph and returns its Proxy.

        Returns:
            Proxy: Proxy for newly created Node.
        """

        return self.proxy_class(Node(*args, graph=self, **kwargs))

    def add(self, node: Node) -> None:
        """Adds a Node to this Graph. Called by Nodes on __init__.

        Args:
            node (Node): Node to add.
        """

        # If we're validating and the user did not provide a value, execute the given target with meta proxy values to compute new proxy_value.
        if self.validate and node.proxy_value is inspect._empty:

            # Enter FakeMode.
            with FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(assume_static_by_default=True),
            ) as fake_mode:
                with FakeCopyMode(fake_mode):

                    proxy_args, proxy_kwargs = Node.prepare_inputs(
                        (node.args, node.kwargs), proxy=True
                    )

                    node.proxy_value = node.target(
                        *proxy_args,
                        **proxy_kwargs,
                    )

        # Get name of target.
        name = node.target if isinstance(node.target, str) else node.target.__name__

        # Init name_idx tracker for this Node's name if not already added.
        if name not in self.name_idx:
            self.name_idx[name] = 0

        # If Node's name is not set, set it to the name_idxed version.
        if node.name is None:
            node.name = f"{name}_{self.name_idx[name]}"

        # Increment name_idx for name.
        self.name_idx[name] += 1

        # Set batch arg to -1
        if getattr(node.target, "attachment_name", False) == "nnsight_module_nodes":
            if self.default:
                # Arg 1 corresponds to batch_size
                node.args[1] = -1

        # Add Node.
        self.nodes[node.name] = node

    def copy(self):
        """Copy constructs a new Graph and then recursively 
        creates new Nodes on the graph.
        """
        new_graph = Graph(validate=False, proxy_class=self.proxy_class)

        def compile(graph, old_node):
            if old_node.name in graph.nodes:
                return graph.nodes[old_node.name]

            # Skip saving node to new graph if it's a .save()
            if old_node.target is LockProtocol:
                return

            node = graph.create(
                target=old_node.target,
                name=old_node.name,
                proxy_value=None,
                args=apply(
                    old_node.args, 
                    lambda x: compile(graph, x), Node
                ),
                kwargs=apply(
                    old_node.kwargs, 
                    lambda x: compile(graph, x), Node
                )
            ).node

            if isinstance(node.target, type) and issubclass(
                node.target, Protocol
            ):
                node.target.compile(node)

            return node

        for node in self.nodes.values():
            compile(new_graph, node)

        return new_graph

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

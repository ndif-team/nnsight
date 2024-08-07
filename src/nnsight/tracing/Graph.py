from __future__ import annotations

import inspect
from typing import Dict, Type

from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
import pygraphviz as pgv

from ..util import apply
from .Node import Node
from .protocols import Protocol
from .Proxy import Proxy


class Graph:
    """Represents a computation graph composed of Nodes


    Attributes:
        validate (bool): If to execute nodes as they are added with their proxy values in order to check if the executions are possible (i.e shape errors etc). Defaults to True.
        sequential (bool): If to run nodes sequentially when executing this graph. Otherwise, only execute root nodes and have them execute nodes downstream.
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
        validate: bool = False,
        sequential: bool = True,
        graph_id: int = None,
    ) -> None:

        self.id = graph_id or id(self)

        self.proxy_class = proxy_class
        self.validate = validate
        self.sequential = sequential

        self.alive = True

        self.nodes: Dict[str, Node] = dict()
        self.name_idx: Dict[str, int] = dict()

        self.attachments = dict()

    def execute(self) -> None:
        """Re-compile graph to prepare for a new execution of the graph.

        Resets all nodes, then compiles all nodes.
        """

        # Reset nodes individually.
        for node in self.nodes.values():
            node.reset()

        if self.sequential:

            for node in self.nodes.values():
                node.execute()

        else:

            root_nodes = [node for node in self.nodes.values() if node.fulfilled()]

            for node in root_nodes:
                node.execute()

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

        # Add Node.
        self.nodes[node.name] = node

    def copy(self):
        """Copy constructs a new Graph and then recursively
        creates new Nodes on the graph.
        """
        new_graph = Graph(
            validate=self.validate,
            sequential=self.sequential,
            proxy_class=self.proxy_class,
        )

        def compile(graph: Graph, old_node: Node):
            if old_node.name in graph.nodes:
                return graph.nodes[old_node.name]

            node = graph.create(
                target=old_node.target,
                name=old_node.name,
                proxy_value=None,
                args=apply(old_node.args, lambda x: compile(graph, x), Node),
                kwargs=apply(old_node.kwargs, lambda x: compile(graph, x), Node),
            ).node

            if isinstance(node.target, type) and issubclass(node.target, Protocol):
                node.target.compile(node)

            return node

        # To preserve order
        nodes = {}

        for node in self.nodes.values():

            compile(new_graph, node)

            # To preserve order
            nodes[node.name] = new_graph.nodes[node.name]

        # To preserve order
        new_graph.nodes = nodes

        return new_graph

    def vis(self, title: str = "graph", path: str = "."):
        """ Generates and saves a graphical visualization of the Intervention Graph using the pygraphviz library. 
        Args:
            title (str): Name of the Intervention Graph. Defaults to "graph".
            path (str): Directory path to save the graphic in. If None saves content to the current directory.
        """

        graph: pgv.AGraph = pgv.AGraph(strict=True, directed=True)

        graph.graph_attr.update(label=title, fontsize='20', labelloc='t', labeljust='c')    
        
        for node in self.nodes.values(): 
            node.visualize(graph)

        graph.draw(f"{path}/{title}.png", prog="dot")

    def __str__(self) -> str:
        result = ""

        for name, node in self.nodes.items():
            result += f"  %{node}\n"

        return result

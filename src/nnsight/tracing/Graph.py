from __future__ import annotations

import inspect
import tempfile
from typing import Callable, Dict, Iterator, List, Optional, Type

from PIL import Image as PILImage

from ..util import apply
from .Node import Node
from .protocols import EarlyStopProtocol
from .Proxy import Proxy
from .util import validate


class Graph:
    """Represents a computation graph composed of :class:`Nodes <nnsight.tracing.Node.Node>`.


    Attributes:
        nodes (Dict[str, :class:`Node <nnsight.tracing.Node.Node>`]): Mapping of `Node` name to node. Order is preserved and important when executing the graph sequentially.
        attachments (Dict[str, Any]): Dictionary object used to add extra functionality to this Graph. Used by Protocols.
        proxy_class (Type[class:`Proxy <nnsight.tracing.Proxy.Proxy>`]): Proxy class to use. Defaults to class:`Proxy <nnsight.tracing.Proxy.Proxy>`.
        alive (bool): If this Graph should be considered alive (still tracing), and therefore added to. Used by `Node`s.
        name_idx (Dict[str, int]): Mapping of node target_name to number of previous names with the same target_name. Used so names are unique.
        
        validate (bool): If to execute nodes as they are added with their proxy values in order to check if the executions are possible and create a new proxy_value. Defaults to True.
        
            When adding `Node`s to the `Graph`, if the `Graph`'s validate attribute is set to `True`, \
            it will execute the `Node`'s target with its arguments' `.proxy_value` attributes (essentially executing the Node, with FakeTensors in FakeTensorMode).
            This 1.) checks to see of the operation is valid on the tensor shape's within the `.proxy_value`s (this would catch an indexing error) and \
            2.) populating this new `Node`'s `.proxy_value` attribute with the result.
        
        sequential (bool): If to run nodes sequentially when executing this graph.
        
            When this is set to `True`, `Node`s attempt to be executed in the order they were added to the `Graph` when calling `.execute(). \
            Otherwise, all nodes are checked to be fulfilled (they have no dependencies). These are root nodes and they are then executed in the order they were added.

        
    """

    def __getstate__(self) -> Dict:
        return {"id": self.id, 
                "nodes": self.nodes, 
                "name_idx": self.name_idx, 
                "sequential": self.sequential}

    def __setstate__(self, state: Dict) -> None:

        self.__dict__.update(state)

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

    def reset(self) -> None:
        """Resets the Graph to prepare for a new execution of the Graph.
        Calls `.reset()` on all Nodes.
        """

        # Reset Nodes individually.
        for node in self.nodes.values():
            node.reset()

    def execute(self) -> None:
        """Executes operations of `Graph`.

        Executes all `Node`s sequentially if `Graph.sequential`. Otherwise execute only root `Node`s sequentially.
        """
        
        self.alive = False

        if self.sequential:
            is_stopped_early: bool = False
            early_stop_execption: Optional[EarlyStopProtocol.EarlyStopException] = None
            for node in self.nodes.values():
                if not is_stopped_early:
                    if node.fulfilled():
                        try:
                            node.execute()
                        except EarlyStopProtocol.EarlyStopException as e:
                            is_stopped_early = True
                            early_stop_execption = e
                            continue
                else:
                    node.clean()
            if is_stopped_early:
                raise early_stop_execption
        else:

            root_nodes = [node for node in self.nodes.values() if node.fulfilled()]

            for node in root_nodes:
                node.execute()

    def create(self, *args, **kwargs) -> Proxy:
        """Creates a Node directly on this `Graph` and returns its `Proxy`.

        Returns:
            Proxy: `Proxy` for newly created `Node`.
        """

        return self.proxy_class(Node(*args, graph=self, **kwargs))

    def add(self, node: Node) -> None:
        """Adds a Node to this Graph. Called by Nodes on __init__.
        
        When adding `Node`s to the `Graph`, if the `Graph`'s validate attribute is set to `True`, \
        it will execute the `Node`'s target with its arguments' `.proxy_value` attributes (essentially executing the Node, with FakeTensors in FakeTensorMode).
        This 1.) checks to see of the operation is valid on the tensor shape's within the `.proxy_value`s (this would catch an indexing error) and \
        2.) populating this new `Node`'s `.proxy_value` attribute with the result.
    

        Args:
            node (Node): Node to add.
        """

        # If we're validating and the user did not provide a proxy_value, execute the given target with meta proxy values to compute new proxy_value.
        if self.validate and node.proxy_value is inspect._empty:

            node.proxy_value = validate(node.target, *node.args, **node.kwargs)

        node_name = self.node_name(node.target)

        if node.name is None:
            node.name = node_name

        # Add Node.
        self.nodes[node.name] = node

    def node_name(self, target: Callable):

        target_name = target.__name__

        # Init name_idx tracker for this Node's name if not already added.
        if target_name not in self.name_idx:
            self.name_idx[target_name] = 0

        node_name = f"{target_name}_{self.name_idx[target_name]}"

        # Increment name_idx for name.
        self.name_idx[target_name] += 1

        return node_name

    def copy(self, graph: Graph = None, return_mapping: bool = False):

        if graph is None:

            graph = Graph(
                validate=self.validate,
                sequential=self.sequential,
                proxy_class=self.proxy_class,
            )

        mapping = {}

        def _copy(node: Node):

            name = node.name

            if name in graph.nodes and name in mapping:
                return graph.nodes[mapping[name]]

            node = graph.create(
                target=node.target,
                proxy_value=None,
                args=apply(node.args, lambda x: _copy(x), Node),
                kwargs=apply(node.kwargs, lambda x: _copy(x), Node),
            ).node

            mapping[name] = node.name

            return node

        nodes = {**graph.nodes}

        for node in self.nodes.values():

            node = _copy(node)

            nodes[node.name] = node

        graph.nodes = nodes

        if return_mapping:
            mapping = {v: k for k, v in mapping.items()}
            return (graph, mapping)

        return graph

    def vis(
        self,
        title: str = "graph",
        path: str = ".",
        display: bool = True,
        save: bool = False,
        recursive: bool = False,
    ):
        """Generates and saves a graphical visualization of the Intervention Graph using the pygraphviz library.
        Args:
            title (str): Name of the Intervention Graph. Defaults to "graph".
            path (str): Directory path to save the graphic in. If None saves content to the current directory.
            display (bool): If True, shows the graph image.
            save (bool): If True, saves the graph to the specified path.
            recursive (bool): If True, recursively visualize sub-graphs.
        """

        try:

            import pygraphviz as pgv

        except Exception as e:

            raise type(e)(
                "Visualization of the Graph requires `pygraphviz` which requires `graphviz` to be installed on your machine."
            ) from e

        from IPython.display import Image
        from IPython.display import display as IDisplay

        graph: pgv.AGraph = pgv.AGraph(strict=True, directed=True)

        graph.graph_attr.update(label=title, fontsize="20", labelloc="t", labeljust="c")

        for node in self.nodes.values():
            # draw bottom up
            if len(node.listeners) == 0:
                node.visualize(graph, recursive)

        def display_graph(file_name):
            in_notebook = True

            # Credit: Till Hoffmann - https://stackoverflow.com/a/22424821
            try:
                from IPython import get_ipython

                if "IPKernelApp" not in get_ipython().config:
                    in_notebook = False
            except ImportError:
                in_notebook = False
            except AttributeError:
                in_notebook = False

            if in_notebook:
                IDisplay(Image(filename=file_name))
            else:
                img = PILImage.open(file_name)
                img.show()
                img.close()

        if not save:
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                graph.draw(temp_file.name, prog="dot")
                if display:
                    display_graph(temp_file.name)
        else:
            graph.draw(f"{path}/{title}.png", prog="dot")
            if display:
                display_graph(f"{path}/{title}.png")

    def __str__(self) -> str:
        result = ""

        for name, node in self.nodes.items():
            result += f"  %{node}\n"

        return result


class MultiGraph(Graph):

    def __init__(self, graphs: List[Graph]) -> None:

        super().__init__()

        self.id_to_graphs = {graph.id: graph for graph in graphs}

        for graph in self:

            self.nodes.update(
                {f"{key}_{graph.id}": value for key, value in graph.nodes.items()}
            )

    def __getstate__(self):
        state = super().__getstate__()
        state["id_to_graphs"] = self.id_to_graphs
        return state

    def __iter__(self) -> Iterator[Graph]:
        return list(self.id_to_graphs.values())

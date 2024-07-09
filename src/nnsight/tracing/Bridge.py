from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from . import protocols

if TYPE_CHECKING:
    from .Graph import Graph


class Bridge:
    """A Bridge object collects and tracks multiple Graphs in order to facilitate interaction between them.
    The order in which Graphs added matters as Graphs can only get values from previous Graphs/

    Attributes:
        id_to_graph (Dict[int, Graph]): Mapping of graph id to Graph.
        locks (int): Count of how many entities are depending on ties between graphs not to be released.
    """

    def __init__(self) -> None:

        # Mapping fro Graph if to Graph.
        self.id_to_graph: Dict[int, "Graph"] = OrderedDict()
        # Stack to keep track of most inner current graph
        self.graph_stack: List["Graph"] = list()

        self.locks = 0

    @property
    def release(self) -> bool:

        return not self.locks

    def add(self, graph: "Graph") -> None:
        """Adds Graph to Bridge.

        Args:
            graph (Graph): Graph to add.
        """

        protocols.BridgeProtocol.set_bridge(graph, self)

        self.id_to_graph[graph.id] = graph

        self.graph_stack.append(graph)

    def peek_graph(self) -> "Graph":
        """Gets the current hierarchical Graph in the Bridge.

        Returns:
            Graph: Graph of current context.

        """
        return self.graph_stack[-1]

    def pop_graph(self) -> None:
        """Pops the last Graph in the graph stack."""

        self.graph_stack.pop()

    def get_graph(self, id: int) -> "Graph":
        """Returns graph from Bridge given the Graph's id.

        Args:
            id (int): Id of Graph to get.

        Returns:
            Graph: Graph.
        """

        return self.id_to_graph[id]

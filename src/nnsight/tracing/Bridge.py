from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

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
        self.id_to_graph: Dict[int, Graph] = OrderedDict()

        self.locks = 0

    @property
    def release(self) -> bool:

        return not self.locks

    def add(self, graph: Graph) -> None:
        """Adds Graph to Bridge.

        Args:
            graph (Graph): Graph to add.
        """

        self.id_to_graph[graph.id] = graph

    def get_graph(self, id: int) -> Graph:
        """Returns graph from Bridge given the Graph's id.

        Args:
            id (int): Id of Graph to get.

        Returns:
            Graph: Graph.
        """

        return self.id_to_graph[id]

    def rank(self, graph: Graph) -> int:
        """Returns rank of Graph. Lower rank means it's been added earlier.

        Args:
            graph (Graph): Graph of rank to get.

        Returns:
            int: Rank of Graph.
        """

        if graph.id not in self.id_to_graph:

            return len(self.id_to_graph)

        return list(self.id_to_graph.keys()).index(graph.id)
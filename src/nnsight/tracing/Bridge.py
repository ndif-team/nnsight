from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union, Optional

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
        # Stack to keep track of most inner current graph
        self._graph_stack: List[Graph] = list()

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
        self._graph_stack.append(graph)

    def peek_graph(self) -> Optional[Graph]:
        """ Gets the current hierarchical Graph in the Bridge.

        Returns:
            Graph: Graph of current context.
        
        """

        if len(self._graph_stack) > 0:
            return self._graph_stack[-1]

    def pop_graph(self) -> None:
        """ Pops the last Graph in the graph stack. """

        if len(self._graph_stack) > 0:
            self._graph_stack.pop()

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

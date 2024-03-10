from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

from .. import util
from . import protocols
from .Graph import Graph
from .Node import Node
from .Proxy import Proxy


class Bridge:

    def __init__(self) -> None:

        self.id_to_graph = OrderedDict()

        self.release = True

    def add(self, graph: Graph) -> None:

        self.id_to_graph[graph.id] = graph

    def get_graph(self, id: int) -> Graph:

        return self.id_to_graph[id]

    def rank(self, graph: Graph) -> int:

        if graph.id not in self.id_to_graph:

            return len(self.id_to_graph)

        return list(self.id_to_graph.keys()).index(graph.id)


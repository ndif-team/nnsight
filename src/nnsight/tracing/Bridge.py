from typing import Any, Dict, List, Tuple, Union

from .Graph import Graph
from .Node import Node


class Bridge:

    def __init__(self, id_to_graph: Dict[int, Graph] = None) -> None:

        self.id_to_graph = id_to_graph or dict()
        
        self.release = True

    def add(self, graph: Graph):

        self.id_to_graph[graph.id] = graph

    def get_graph(self, id: int) -> Graph:

        return self.id_to_graph[id]
    

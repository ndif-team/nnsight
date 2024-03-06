from .Graph import Graph
from .Node import Node
from typing import Any, List
class Bridge:
    
    
    def __init__(self, graphs: List[Graph] = None) -> None:
        
        self.graphs = graphs or []
        
        
    def add_bridge_node(self, graph: Graph):
        
        pass
    
    
    def __call__(self, node: Node) -> Any:
        
        return self.graphs[graph_idx].nodes[]

from typing import Any
from . import Backend

from ...tracing import protocols
from ...tracing import Graph


class GraphBackend(Backend):
    
    def __init__(self, graph: Graph) -> None:
        
        self.graph = graph
    
    
    def __call__(self, obj: protocols.Protocol) -> None:
        return super().__call__(obj)
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, List, Tuple
from ...tracing.Graph import Graph
from .Collection import Collection
if TYPE_CHECKING:
    from .Accumulator import Accumulator

class Iterator(AbstractContextManager, Collection):
    
    def __init__(self, accumulator: "Accumulator") -> None:
        
        self.accumulator = accumulator
        
        self.graph: Graph = Graph(
            self.accumulator.model, proxy_class=self.accumulator.model.proxy_class, validate=validate
        )

        
        self.collection = [] 

    def __enter__(self) -> Iterator:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
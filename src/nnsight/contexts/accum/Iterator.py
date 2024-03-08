from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Iterable

from ...tracing.Graph import Graph
from ..Tracer import Executable
from ..backends import LocalMixin
from .Collector import Collection
if TYPE_CHECKING:
    from .Accumulator import Accumulator


class Iterator(AbstractContextManager, Collection, Executable):

    def __init__(self, accumulator: "Accumulator", data: Iterable) -> None:

        self.accumulator = accumulator
        self.data = data


    def __enter__(self) -> Iterator:
        pass
        
        

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
    
    
    ### BACKENDS ########
    
    def local_backend_execute(self) -> None:
        
        
        for item in self.data:
            pass
            
            
            
        
        return super().local_backend_execute()

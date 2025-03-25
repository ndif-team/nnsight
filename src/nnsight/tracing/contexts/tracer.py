from typing import Callable, TypeVar, Union
from typing_extensions import Self

from ..graph import ProxyType, SubGraph, NodeType, Proxy
from ..protocols import StopProtocol
from . import Condition, Context, Iterator

class Tracer(Context[SubGraph[NodeType, ProxyType]]):
    
    
    def __enter__(self) -> Self:
        
        from .globals import GlobalTracingContext
        
        GlobalTracingContext.try_register(self)
        
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        
        from .globals import GlobalTracingContext
        
        GlobalTracingContext.try_deregister(self)
        
        return super().__exit__(exc_type, exc_val, exc_tb)

    def iter(self, collection):

        return Iterator(collection, parent=self.graph)

    def cond(self, condition):

        return Condition(condition, parent=self.graph)
    
    def stop(self):

        StopProtocol.add(self.graph)
        
    def log(self, *args):
        
        self.apply(print, *args)

    R = TypeVar('R')
    
    def apply(self, target: Callable[..., R], *args, **kwargs) -> Union[Proxy, R]:

        return self.graph.create(
            target,
            *args,
            **kwargs,
        )

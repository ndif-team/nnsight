from typing import Callable, Any, Generic
from typing_extensions import Self

from ..graph import ProxyType, GraphType, SubGraph, NodeType, Proxy
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

    def trace(self):

        return Tracer(parent=self.graph)

    def iter(self, collection):

        return Iterator(collection, parent=self.graph)

    def cond(self, condition):

        return Condition(condition, parent=self.graph)

    def apply(
        self,
        target: Callable,
        *args,
        **kwargs,
    ) -> Proxy:

        return self.graph.create(
            target,
            *args,
            **kwargs,
        )

    def stop(self):

        StopProtocol.add(self.graph)

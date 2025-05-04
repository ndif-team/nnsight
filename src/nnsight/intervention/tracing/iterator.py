from typing import Callable, TYPE_CHECKING, Any, Union
from .invoker import Invoker
from ..interleaver import Mediator

if TYPE_CHECKING:
    from .tracer import InterleavingTracer
else:
    InterleavingTracer = Any
    
class IteratorProxy:
    
    def __init__(self, tracer: InterleavingTracer):
        self.tracer = tracer
        
    def __getitem__(self, iteration: Union[int, slice]):
        return IteratorTracer(iteration, self.tracer)
    
class IteratorTracer(Invoker):
    
    def __init__(self, iteration: Union[int, slice], *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.iteration = iteration
        
        
    def execute(self, fn: Callable):
                
        mediator = Mediator(fn, self.info)
        
        self.tracer.model._interleaver.iter(mediator, self.iteration)
    
    
    
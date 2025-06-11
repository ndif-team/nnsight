from typing import Callable, TYPE_CHECKING, Any, Union
from .invoker import Invoker
from ..interleaver import Mediator

if TYPE_CHECKING:
    from .tracer import InterleavingTracer
else:
    InterleavingTracer = Any
    
class IteratorProxy:
    
    def __init__(self, interleaver):
        self.interleaver = interleaver
        
    def __getitem__(self, iteration: Union[int, slice]):
        return IteratorTracer(iteration, self.interleaver, None)
    
class IteratorTracer(Invoker):
    
    def __init__(self, iteration: Union[int, slice], interleaver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.interleaver = interleaver
        
        self.iteration = iteration
        
        
    def execute(self, fn: Callable):
                
        mediator = Mediator(fn, self.info)
        mediator.name = "Iterator" + mediator.name
        
        self.interleaver.current.iter(mediator, self.iteration)
    
    
    
from typing import Callable, TYPE_CHECKING, Any
import torch
from .invoker import Invoker

from ...util import Patch
from ..interleaver import Mediator

if TYPE_CHECKING:
    from ..interleaver import Interleaver
else:
    Interleaver = Any
    
class BackwardsTracer(Invoker):
    
    def __init__(self, tensor: torch.Tensor, fn: Callable, interleaver: Interleaver, *args, **kwargs):

        super().__init__(None, *args, **kwargs)
        
        self.tensor = tensor
        self.fn = fn
        self.interleaver = interleaver        
                
    def execute(self, fn: Callable):
                
        mediator = Mediator(fn, self.info) 
        mediator.name = "Backwards" + mediator.name
        
        grad_patch = Patch(torch.Tensor, self.interleaver.wrap_grad(), "grad")
        
        self.interleaver.patcher.add(grad_patch)
        
        def inner():
                        
            self.fn(self.tensor, *self.args, **self.kwargs)
                        
            grad_patch.restore()
                 
        self.interleaver.current.register(mediator, inner)
import ast
from typing import Callable, TYPE_CHECKING, Any
import torch
from .invoker import Invoker

from .util import try_catch
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
                 
        self.interleaver.register(mediator, lambda: self.fn(self.tensor, *self.args, **self.kwargs))
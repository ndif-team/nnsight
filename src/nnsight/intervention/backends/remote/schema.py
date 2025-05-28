from pydantic import BaseModel
from typing import Callable, TYPE_CHECKING
from typing import Any  
from .unpickler import RemoteUnpickler
from .utils import Protector, WHITELISTED_MODULES_DESERIALIZATION
import dill

if TYPE_CHECKING:
    from ...tracing.tracer import Tracer
    from ...envoy import Envoy
    
    
else:
    Tracer = Any

class Request(BaseModel):
    model_key: str
    intervention:Callable | None = None
    tracer:Tracer
    
    @classmethod
    def from_pickle(cls, file):
        
        dill.settings['trace'] = True
        dill.settings['recurse'] = True
        
            
        with Protector(WHITELISTED_MODULES_DESERIALIZATION):
            request = RemoteUnpickler(file).load()
            
        return request
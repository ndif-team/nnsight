from .base import Backend
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any
    
class EditingBackend(Backend):
    
    
    def __call__(self, tracer: Tracer):
        
        tracer.model._default_source = tracer.info.source
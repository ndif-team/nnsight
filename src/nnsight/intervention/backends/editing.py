from .base import Backend
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from ..tracing.tracer import InterleavingTracer
else:
    InterleavingTracer = Any
    
class EditingBackend(Backend):
    
    
    def __call__(self, tracer: InterleavingTracer):
                
        tracer.model._default_source = tracer.model._default_source + [tracer.info.source]
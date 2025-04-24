from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any
    

class Backend:
    
    def __call__(self, tracer: Tracer):
        raise NotImplementedError("Subclasses must implement this method")
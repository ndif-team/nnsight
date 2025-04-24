from .base import Backend
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any
    
class ExecutionBackend(Backend):
    
    
    
    def __call__(self, tracer: Tracer):
        
        tracer.compile()
        
        source = "".join(tracer.info.source)
        
        local_namespace = {}
        
        # Execute the function definition in the local namespace
        exec(source, {**tracer.info.frame.f_globals, **tracer.info.frame.f_locals}, local_namespace)
        
        fn = list(local_namespace.values())[-1]
                
        tracer.execute(fn)
from .base import Backend
from typing import TYPE_CHECKING, Any
from ..tracing.util import wrap_exception
if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any
    
class ExecutionBackend(Backend):
    
    def __call__(self, tracer: Tracer):
        
        tracer.compile()
        
        source = "".join(tracer.info.source)
                
        filename = "<nnsight>"
        
        code_obj = compile(source, filename, 'exec')
        
        local_namespace = {}
        
        # Execute the function definition in the local namespace
        exec(code_obj, {**tracer.info.frame.f_globals, **tracer.info.frame.f_locals}, local_namespace)
        
        fn = list(local_namespace.values())[-1]
        
        # TODO maybe move it tracer __exit__
        try:
            tracer.execute(fn)
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
       
                
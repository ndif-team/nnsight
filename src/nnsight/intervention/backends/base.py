from builtins import compile, exec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any
    

class Backend:
    
    def __call__(self, tracer: Tracer):
        
        has_globals =  tracer.info.source[0].lstrip().startswith('global')
        
        tracer.compile()
        
        if has_globals:
            tracer.info.start_line -= 1

        source = "".join(tracer.info.source)
                
        code_obj = compile(source, tracer.info.filename, "exec")
        
        local_namespace = {}

        # Execute the function definition in the local namespace
        exec(
            code_obj,
            {**tracer.info.frame.f_globals, **tracer.info.frame.f_locals},
            local_namespace,
        )

        fn = list(local_namespace.values())[-1]
        
        return fn
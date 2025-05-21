import inspect
from ..base import Backend
from .. import ExecutionBackend
from typing import TYPE_CHECKING, Any
from .schema import Request, TracerTypes
from .utils import Protector, ProtectorEscape, whitelisted_modules
if TYPE_CHECKING:
    from ...tracing.tracer import Tracer
else:
    Tracer = Any
    
class RemoteBackend(Backend):
    
    def __init__(self, model:Any):
        self.model = model
    
    def __call__(self, tracer: Tracer):
        
        request = Request(
            model_var_name = tracer.info.node.items[0].context_expr.func.value.id,
            source = tracer.info.source,
            args=(tracer.args, tracer.kwargs),
            variables = {},
            tracer_type = TracerTypes.TRACER,
            fn=tracer.fn.__name__
        )
        
        # from .DependencyCollector import DependencyCollector
        
        # collector = DependencyCollector()
        
        # variables, functions = collector.collect_dependencies_from_with_block(tracer.info.node)

        from ...tracing.tracer import InterleavingTracer, Tracer
        
        info = Tracer.Info(
            source = request.source,
            frame = None,
            node = None,
            start_line = 0,
        )
        
        fn = getattr(self.model, request.fn)
        
        args, kwargs = request.args
        
        tracer = InterleavingTracer(fn, self.model, *args, **kwargs, _info=info)
        
        RemoteExecutionBackend(self.model, request.model_var_name)(tracer)
        
        
class RemoteExecutionBackend(Backend):
    
    def __init__(self, model:Any, model_var_name:str):
        self.model = model
        self.model_var_name = model_var_name
        
    def __call__(self, tracer: Tracer):
        
        
        protector = Protector(whitelisted_modules)
        escape = ProtectorEscape(protector)

        def execute():
            
            frame = inspect.currentframe()
            
            frame.f_locals[self.model_var_name] = self.model
            
            tracer.info.frame = frame
          
            with protector:
                with escape:    
                    
                    ExecutionBackend()(tracer)
                
        execute()
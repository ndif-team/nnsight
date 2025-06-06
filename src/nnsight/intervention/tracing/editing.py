import ast
from .tracer import InterleavingTracer
from ..backends.editing import EditingBackend
from ..backends.base import Backend

class EditingTracer(InterleavingTracer):
    
    def __init__(self, *args, backend:Backend = EditingBackend(), inplace:bool=False, **kwargs):
        
        self.capture()
        
        self.return_tracer = False
        
        if self.info.node.items[0].optional_vars is not None:

            if isinstance(self.info.node.items[0].optional_vars, ast.Tuple):

                self.return_tracer = True
                
                self.model_var_name = self.info.node.items[0].optional_vars.elts[0].id
                self.tracer_var_name = self.info.node.items[0].optional_vars.elts[1].id
                
            else:
                
                self.model_var_name = self.info.node.items[0].optional_vars.id
                self.tracer_var_name = "__nnsight_tracer__"
        
        
        super().__init__(*args, backend=backend, **kwargs)
        
        if not inplace:
            self.model = self.model._shallow_copy()  
    
    def __enter__(self):
        
        super().__enter__()
        
        if self.return_tracer:
            return self.model, self
        
        return self.model

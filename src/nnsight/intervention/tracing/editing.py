import ast
from .tracer import InterleavingTracer
from ..backends.editing import EditingBackend
from ..backends.base import Backend

class EditingTracer(InterleavingTracer):
    
    def __init__(self, *args, backend:Backend = EditingBackend(), inplace:bool=False, **kwargs):
        
        self.capture()
        
        self.return_tracer = False
        
        super().__init__(*args, backend=backend, **kwargs)
        
        if not inplace:
            self.model = self.model._shallow_copy()  
    
    def __enter__(self):
        
        super().__enter__()
        
        if self.return_tracer:
            return self.model, self
        
        return self.model

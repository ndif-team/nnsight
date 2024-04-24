class IteratorMixin:
    
    def iterator_backend_execute(self, release:bool = False):
        
        raise NotImplementedError()

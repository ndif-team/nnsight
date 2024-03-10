class IteratorMixin:
    
    def iterator_backend_execute(self, last_iter:bool = False):
        
        raise NotImplementedError()

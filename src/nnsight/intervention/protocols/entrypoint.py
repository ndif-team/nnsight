from ...tracing.protocols import Protocol

class EntryPoint(Protocol):
    
    @classmethod
    def add(cls,*args, **kwargs):
        return super().add(*args, redirect=False, **kwargs)

from typing import Any
from ...tracing.protocols import Protocol

class EntryPoint(Protocol):
    """An EntryPoint Protocol should have its value set manually outside of normal graph execution.
    This makes these type of Nodes special and are handled differently in a variety of cases.
    Subclasses EntryPoint informs those cases to handle it differently.
    Examples are InterventionProtocol and GradProtocol.
    """
    
    @staticmethod
    def is_entrypoint(thing:Any):
        
        return isinstance(thing, type) and issubclass(thing, EntryPoint)
    
    @classmethod
    def add(cls,*args, **kwargs):
        return super().add(*args, redirect=False, **kwargs)

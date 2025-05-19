import contextlib
from typing import Any
from ..._c.py_mount import mount



class Wrapper:
    
    saves = set()
    
    
    @staticmethod
    def save(obj: Any):
        
        Wrapper.saves.add(id(obj))
        
        return obj
        
        
mount(Wrapper.save, "save")
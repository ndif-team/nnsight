from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..graph import Graph, Node, Proxy
    
class Protocol:


    @classmethod
    def add(cls, graph:"Graph",*args, **kwargs) -> "Proxy":
        
        return graph.create(
            cls,
            *args,
            trace_value=None,
            **kwargs
            
        )

    @classmethod
    def execute(cls, node: "Node"):
      
        pass


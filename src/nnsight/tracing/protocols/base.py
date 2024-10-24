from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from ..graph import Graph, Node, Proxy, GraphType, ProxyType
    
class Protocol:

    @classmethod
    def add(cls, graph:"GraphType",*args, **kwargs) -> "Proxy":
        
        return graph.create(
            cls,
            *args,
            **kwargs
            
        )

    @classmethod
    def execute(cls, node: "Node"):
      
        pass


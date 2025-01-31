from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..graph import GraphType, Node, Proxy
    
class Protocol:
    
    @staticmethod
    def is_protocol(thing:Any):
        
        return isinstance(thing, type) and issubclass(thing, Protocol)

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

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "black", "shape": "ellipse"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: {"style": "solid"}),
        }  # Argument edge display

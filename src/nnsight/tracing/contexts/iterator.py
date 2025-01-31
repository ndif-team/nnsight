import copy
from typing import Collection, Dict, Any

from ...tracing.graph import SubGraph
from ...tracing.graph import Node
from ...tracing.graph import Proxy
from . import Context
from ..protocols import VariableProtocol, StopProtocol

class Iterator(Context[SubGraph]):

    def __init__(self, collection: Collection, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.args = [collection]
        
    def __enter__(self) -> Proxy:
        
        super().__enter__()
        
        return VariableProtocol.add(self.graph)
        

    @classmethod
    def execute(cls, node: Node):

        graph, collection = node.args

        graph: SubGraph
        collection: Collection

        collection = node.prepare_inputs(collection)
        
        variable_node = next(iter(graph))
        
        graph.defer_stack.append(variable_node.index)

        for idx, value in enumerate(copy.copy(collection)):

            VariableProtocol.set(variable_node, value)

            if idx == len(collection) - 1:
                graph.defer_stack.pop()


            graph.reset()
            try:
                graph.execute()
            except Exception as e:
                
                if idx != len(collection) - 1:
                
                    graph.defer_stack.pop()
                    
                if not isinstance(e, StopProtocol.StopException):
                
                    raise e
                
                else:
                    break
            
        node.set_value(None)

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        default_style = super().style()
        
        default_style["node"] = {"color": "blue", "shape": "polygon", "sides": 6}

        return default_style

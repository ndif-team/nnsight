from typing import Any, Dict
from pydantic import BaseModel, ConfigDict
from ..tracing.graph import Graph
RESULT = Dict[int, Any]

class ResultModel(BaseModel):
    
    
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
    
    id:str
    result: RESULT
    
    @classmethod
    def inject(cls, graph:Graph, result:RESULT):
    
                
        for index, value in result.items():
            
            graph.nodes[index]._value = value
            
    @classmethod
    def from_graph(cls, graph:Graph) -> RESULT:
        

        return {node.index: node.value for node in graph.nodes if node.done}
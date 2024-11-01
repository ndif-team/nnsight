from typing import TYPE_CHECKING
from nnsight.tracing.contexts.base import Context
from ...tracing.backends import Backend

from ..graph import InterventionGraph
if TYPE_CHECKING:
    from .. import NNsight

class EditingBackend(Backend):
    
    def __init__(self, model: "NNsight") -> None:
        
        self.model = model
    
    def __call__(self, graph: InterventionGraph) -> None:
                
        self.model._default_graph = graph.nodes[-1].args[0]
        
        graph.nodes = []
        graph.stack = []
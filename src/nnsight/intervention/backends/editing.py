from typing import TYPE_CHECKING
from ...tracing.backends import Backend

from ...tracing.graph import Graph
if TYPE_CHECKING:
    from .. import NNsight

class EditingBackend(Backend):
    """Backend to set the default graph to the current InterventionGraph. Assumes the final Node is an InterleavingTracer.
    """
    
    def __init__(self, model: "NNsight") -> None:
        
        self.model = model
    
    def __call__(self, graph: Graph) -> None:
                        
        self.model._default_graph = graph.nodes[-1].args[0]
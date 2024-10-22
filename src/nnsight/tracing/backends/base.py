from ..graph import GraphType
from ..protocols import StopProtocol


class Backend:

    def __call__(self, graph: GraphType) -> None:

        raise NotImplementedError()


class ExecutionBackend(Backend):

    def __call__(self, graph: GraphType) -> None:
        
        try:
         
            graph.nodes[-1].execute()
            
        except StopProtocol.StopException:
            
            pass
        
        finally:
                
            graph.nodes = []
            graph.stack = []
            

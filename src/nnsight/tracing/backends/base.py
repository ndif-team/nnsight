from ..graph import Graph
from ..protocols import StopProtocol


class Backend:

    def __call__(self, graph: Graph) -> None:

        raise NotImplementedError()


class ExecutionBackend(Backend):

    def __call__(self, graph: Graph) -> None:

        try:

            graph.nodes[-1].execute()

        except StopProtocol.StopException:

            pass

        finally:

            graph.nodes.clear()
            graph.stack.clear()

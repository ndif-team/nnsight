from ...tracing.graph.graph import Graph
from ...tracing.backends import Backend


class NoopBackend(Backend):

    def __call__(self, graph: Graph) -> None:
        graph.nodes.clear()
        graph.stack.clear()
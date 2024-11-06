from ...tracing.contexts import Tracer
from ...tracing.graph import GraphType, NodeType
from ..protocols import EntryPoint, NoopProtocol

class LocalContext(Tracer):

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        start = self.graph[0].index
        end = self.graph[-1].index

        for node in self.graph.nodes[start : end + 1]:

            for dependency in node.dependencies:

                if (
                    isinstance(dependency.target, type)
                    and issubclass(dependency.target, EntryPoint)
                ) or dependency.index < start:

                    self.args.append(dependency)

        return super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def prune(cls, node: NodeType):

        graph: GraphType = node.args[0]

        start = graph[0].index
        end = graph[-1].index

        for node in graph.nodes[start : end + 1]:

            if isinstance(node.target, type) and issubclass(node.target, EntryPoint):
                continue

            node.target = NoopProtocol

            node.args.clear()
            node.kwargs.clear()
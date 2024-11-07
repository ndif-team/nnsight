from typing import Callable, List, Optional

from nnsight.tracing.graph.node import Node

from ...tracing.contexts import Tracer
from ...tracing.graph import GraphType, NodeType
from ..protocols import EntryPoint, NoopProtocol


class LocalContext(Tracer):

    send: Optional[Callable] = None

    @classmethod
    def set(cls, fn: Callable):

        cls.send = fn

    @classmethod
    def execute(cls, node: NodeType):

        super().execute(node)

        uploads = node.kwargs.get("uploads", [])

        if uploads:

            values = {index: node.graph.nodes[index].value for index in uploads}

            cls.send(values)

            for index in uploads:

                node = node.graph.nodes[index]

                node.remaining_listeners -= 1

                if node.redundant:
                    node.destroy()


class RemoteContext(Tracer):

    send: Optional[Callable] = None
    receive: Optional[Callable] = None

    @classmethod
    def set(cls, send: Callable, receive: Callable):

        cls.send = send
        cls.receive = receive

    @classmethod
    def from_local(cls, local_node: NodeType):

        local_node.target = RemoteContext

        graph: GraphType = local_node.args[0]

        start = graph[0].index
        end = graph[-1].index

        uploads = []

        # TODO check for swap and error

        for node in graph.nodes[start : end + 1]:

            for dependency in node.dependencies:

                if (
                    isinstance(dependency.target, type)
                    and issubclass(dependency.target, EntryPoint)
                ) or dependency.index < start:

                    local_node.args.append(dependency)

            if isinstance(node.target, type) and issubclass(node.target, EntryPoint):
                continue

            node.args.clear()
            node.kwargs.clear()

            node.target = NoopProtocol

            for listener in node.listeners:

                if listener.index > end:

                    uploads.append(node.index)

        if len(uploads) > 0:
            local_node.kwargs["upload"] = True

        return uploads

    @classmethod
    def execute(cls, node: NodeType):

        graph, *dependencies = node.args

        dependencies = {
            dependency.index: dependency.value for dependency in dependencies
        }

        cls.send((node.index, dependencies))

        super().execute(node)

        if node.kwargs.get("upload", False):

            values = cls.receive()

            for index, value in values.items():
                graph.nodes[index]._value = value

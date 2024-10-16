from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from nnsight04.tracing.graph import Graph

from ..backends import ChildBackend
from ..contexts import Context

if TYPE_CHECKING:
    from ...tracing.graph.graph import Graph
    from ...tracing.graph.node import Node


class Condition(Context):

    def __init__(self, condition: Any, branch: "Node" = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.args = [condition, branch]

    def else_(self, condition: Optional[Any] = None):
        
        return Condition(
            condition, branch=self.graph.nodes[self.graph[-1].index+1], backend=ChildBackend(), parent=self.graph.stack[-1]
        )

    @classmethod
    def execute(cls, node: "Node"):
        graph, condition, branch = node.args

        graph: "Graph"

        condition: Any
        condition, branch = node.prepare_inputs((condition, branch))

        if condition is None and not branch:
            condition = True
            
        if not branch and condition:
            graph.reset()
            graph.execute()

            node.set_value(True)
        else:
            cls.clean(graph)
            node.set_value(branch)

    @classmethod
    def clean(cls, graph: "Graph"):

        for node in graph:
            node.update_dependencies()

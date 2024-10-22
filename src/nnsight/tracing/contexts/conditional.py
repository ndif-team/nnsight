from __future__ import annotations

from typing import Any, Optional

from ...tracing.graph import NodeType, ProxyType, SubGraph
from ..contexts import Context


class Condition(Context[SubGraph]):

    def __init__(
        self, condition: Any, branch: NodeType = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.args = [condition, branch]

    def else_(self, condition: Optional[Any] = None):

        return Condition(
            condition,
            branch=self.graph.nodes[self.graph[-1].index + 1],
            parent=self.graph.stack[-1],
        )

    @classmethod
    def execute(cls, node: NodeType):
        graph, condition, branch = node.args

        graph: SubGraph

        condition: Any
        condition, branch = node.prepare_inputs((condition, branch))

        if condition is None and not branch:
            condition = True

        if not branch and condition:
            graph.reset()
            graph.execute()

            node.set_value(True)
        else:
            graph.clean()
            node.set_value(branch)
            



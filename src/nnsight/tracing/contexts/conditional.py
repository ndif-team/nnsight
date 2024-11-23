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
        self.index = None

    def else_(self, condition: Optional[Any] = None):
        
        return Condition(
            condition,
            branch=self.graph.nodes[self.index],
            parent=self.graph.stack[-1],
        )
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        
        self.index = self.graph.nodes[-1].index

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
            



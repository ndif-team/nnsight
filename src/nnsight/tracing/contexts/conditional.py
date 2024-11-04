from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

from ...tracing.graph import NodeType, SubGraph
from ..contexts import Context


class Condition(Context[SubGraph]):

    def __init__(
        self, condition: Optional[NodeType], branch: Optional[NodeType] = None, *args, **kwargs
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

        # else case has a True condition
        if condition is None and not branch:
            condition = True

        if not branch and condition:
            
            graph.reset()
            graph.execute()

            node.set_value(True)
        else:
            graph.clean()
            node.set_value(branch)
            
    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {
                "color": "#FF8C00",
                "shape": "polygon",
                "sides": 6,
            },  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: {"style": "solid"}, {2: {"style": "solid", "label": "branch", "color": "#FF8C00", "fontsize": 10}}), # Argument edge display
        }


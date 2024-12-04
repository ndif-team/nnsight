from typing import TYPE_CHECKING, Any, Dict

from ...tracing.protocols import Protocol

if TYPE_CHECKING:
    from ..graph import InterventionNodeType


class SwapProtocol(Protocol):
    

    @classmethod
    def execute(cls, node: "InterventionNodeType") -> None:
        
        intervention_node, value = node.args
        intervention_node: "InterventionNodeType"
        
        value = node.prepare_inputs(value)
        
        node.set_value(None)
        
        intervention_node.kwargs['swap'] = value

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        default_style = super().style()

        default_style["node"] = {"color": "green4", "shape": "ellipse"}

        return default_style

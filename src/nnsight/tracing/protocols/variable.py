from typing import TYPE_CHECKING, Any, Dict

from . import Protocol

if TYPE_CHECKING:
    from ..graph import Node

class VariableProtocol(Protocol):

    @classmethod
    def set(cls, node: "Node", value: Any):

        node.args = [value]

    @classmethod
    def execute(cls, node: "Node"):

        value = node.prepare_inputs(node.args[0])

        node.set_value(value)

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """ Visualization style for this protocol node.
        
        Returns:
            - Dict: dictionary style.
        """

        default_style = super().style()

        default_style["node"] = {"color": "blue", "shape": "box"}
        
        return default_style

from collections import defaultdict
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

        return {
            "node": {"color": "blue", "shape": "box"}, # Node display
            "label": cls.__name__,
            "arg": defaultdict(lambda: {"color": "gray", "shape": "box"}), # Non-node argument  
            "arg_kname": defaultdict(lambda: None), # Argument label key word
            "edge": defaultdict(lambda: {"style": "solid"}) # Argument edge display
        }

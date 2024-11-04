from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict

from . import Protocol

if TYPE_CHECKING:
    from ..graph import Node


class StopProtocol(Protocol):

    class StopException(Exception):
        pass

    @classmethod
    def execute(cls, node: "Node") -> None:

        raise cls.StopException()
    
    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {
                "color": "red",
                "shape": "polygon",
                "sides": 6,
            },  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: {"style": "solid"}),
        }  # Argument edge display

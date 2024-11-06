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

        default_style = super().style()

        default_style["node"] = {"color": "red", "shape": "polygon", "sides": 6}

        return default_style

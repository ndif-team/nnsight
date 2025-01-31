from typing import TYPE_CHECKING, Any, Dict

from . import Protocol

if TYPE_CHECKING:
    from ..graph import ProxyType, NodeType


class LockProtocol(Protocol):

    @classmethod
    def add(cls, node: "NodeType") -> "ProxyType":
        return node.create(
            cls,
            node,
            fake_value=None,
        )
    
    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        default_style = super().style()

        default_style["node"] = {"color": "brown", "shape": "ellipse"}

        return default_style

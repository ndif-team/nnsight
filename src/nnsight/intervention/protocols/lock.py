from typing import TYPE_CHECKING, Any, Dict

from ...tracing.protocols import Protocol

if TYPE_CHECKING:
    from ..graph import InterventionNodeType, InterventionProxyType


class LockProtocol(Protocol):

    @classmethod
    def add(cls, node: "InterventionNodeType") -> "InterventionProxyType":
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

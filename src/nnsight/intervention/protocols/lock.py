from typing import TYPE_CHECKING, Dict, Any

from collections import defaultdict

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

        """ return {
            "node": {"color": "brown", "shape": "ellipse"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument lable key word
            "edge": defaultdict(lambda: {"style": "solid"}), # Argument edge display
        } """

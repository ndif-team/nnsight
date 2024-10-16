from ...tracing.graph import Node
from ...tracing.protocols import Protocol
from 


class LockProtocol(Protocol):
    """Simple Protocol who's .execute() method does nothing. This means not calling .set_value() on the Node, therefore  the Node won't be destroyed."""

    @classmethod
    def add(cls, node: "Node") -> "InterventionProxy":

        return node.create(
            proxy_value=None,
            target=cls,
            args=[node],
        )

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "brown", "shape": "ellipse"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument lable key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


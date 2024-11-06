from typing import TYPE_CHECKING, Any, Callable
from ...tracing.protocols import Protocol
if TYPE_CHECKING:
    from ..graph import InterventionNode
class StreamingDownloadProtocol(Protocol):

    @classmethod
    def execute(cls, node: "InterventionNode"):
        """When executing remotely, the local version of this Node type has its value set directly by `RemoteBackend`, not via `.execute(...)`
        The remote version streams the value in a ResponseModel object.

        Is a no-op when not executing remotely.
        """

        value_node = node.args[0]

        node.set_value(value_node.value)


class StreamingUploadProtocol(Protocol):

    send: Callable = None

    @classmethod
    def set(cls, fn: Callable):

        cls.send = fn

    @classmethod
    def add(cls, graph: "Graph", value: Any) -> "InterventionProxy":
        """Add streaming upload Node to the intervention graph.

        Args:
            graph (Graph): Graph to add Node to.
            value (Any): Value to upload remotely when available locally.
        """

        return graph.create(target=cls, proxy_value=None, args=[value])

    @classmethod
    def execute(cls, node: "Node"):
        """When executing remotely, the local version of this Node calls `cls.send` to upload the its value to a waiting remote service.
        The remote version blocks and waits until it receives the value from its local counterpart.

        Is a no-op when not executing remotely.

        Args:
            node (Node): Node to upload remotely.
        """

        value = node.prepare_inputs(node.args[0])

        if cls.send is not None:

            cls.send(value)

            node.update_dependencies()

        else:

            node.set_value(value)

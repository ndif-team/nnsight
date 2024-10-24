from typing import TYPE_CHECKING

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
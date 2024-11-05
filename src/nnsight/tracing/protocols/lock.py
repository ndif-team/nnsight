from typing import TYPE_CHECKING

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
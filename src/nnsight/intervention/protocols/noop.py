from typing import TYPE_CHECKING, Any

from ...tracing.protocols import Protocol
if TYPE_CHECKING:
    from ..graph import  InterventionNode


class NoopProtocol(Protocol):
    

    @classmethod
    def execute(cls, node: "InterventionNode") -> None:
    
        node.set_value(None)
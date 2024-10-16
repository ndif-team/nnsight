from typing import TYPE_CHECKING, Any

from . import Protocol

if TYPE_CHECKING:
    from ..graph import Node

class VariableProtocol(Protocol):

    @classmethod
    def set(cls, node: "Node", value: Any):

        node.args = [value]

    @classmethod
    def execute(cls, node: "Node"):

        value = node.prepare_inputs(node.args[0])

        node.set_value(value)

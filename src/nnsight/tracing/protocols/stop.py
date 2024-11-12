from typing import TYPE_CHECKING

from . import Protocol
from ...util import NNsightError

if TYPE_CHECKING:
    from ..graph import Node


class StopProtocol(Protocol):

    class StopException(NNsightError):
        pass

    @classmethod
    def execute(cls, node: "Node") -> None:

        raise cls.StopException("Early Stop Exception!", node.index)


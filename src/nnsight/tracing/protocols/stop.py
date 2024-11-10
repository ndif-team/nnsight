from typing import TYPE_CHECKING

from . import Protocol

if TYPE_CHECKING:
    from ..graph import Node


class StopProtocol(Protocol):

    class StopException(Exception):
        pass

    @classmethod
    def execute(cls, node: "Node") -> None:

        raise cls.StopException()


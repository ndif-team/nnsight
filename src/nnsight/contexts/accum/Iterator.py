from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Tuple

from ...tracing.Graph import Graph
from ..backends import LocalMixin
from .Collector import Collection

if TYPE_CHECKING:
    from .Accumulator import Accumulator


class Iterator(Collection):

    def __init__(self, data: Iterable, *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.data = data
        
        self.graph = Graph()

    def __enter__(self) -> Iterator:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    ### BACKENDS ########

    def local_backend_execute(self) -> None:

        for item in self.data:
            pass

        return super().local_backend_execute()

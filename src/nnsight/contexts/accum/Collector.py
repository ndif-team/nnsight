from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from ..backends import LocalMixin
from ..Tracer import Executable

if TYPE_CHECKING:
    from .Accumulator import Accumulator


class Collection(LocalMixin):

    def __init__(self, accumulator: Accumulator) -> None:
        
        self.accumulator = accumulator
        
        self.collection: List[Executable] = []

    ### BACKENDS ########
    
    def local_backend_execute(self) -> None:

        for executable in self.collection:

            executable.local_backend_execute()
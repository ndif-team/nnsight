from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from ..backends import LocalMixin


class Collection:

    def __init__(self) -> None:

        self.collection: List[LocalMixin] = []

    def __call__(self) -> Any:
        
        for executable in self.collection:
            
            executable.local_backend_execute()

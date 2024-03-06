from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

from .Iterator import Iterator
from ...tracing.Graph import Graph
from .Collection import Collection

if TYPE_CHECKING:
    from ...models.NNsightModel import NNsight


class Accumulator(AbstractContextManager, Collection):

    def __init__(
        self,
        model: "NNsight",
    ) -> None:
        
        self.model = model

    def __enter__(self) -> Accumulator:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def iter(self, iterable) -> Iterator:

        return Iterator(self, iterable)

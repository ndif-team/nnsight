from __future__ import annotations
from contextlib import AbstractContextManager

from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from ..backends import AccumulatorMixin, Backend, LocalMixin
from ..Tracer import Tracer

if TYPE_CHECKING:
    from .Accumulator import Accumulator


class Collection(AbstractContextManager, LocalMixin, AccumulatorMixin):

    def __init__(self, backend: Backend, accumulator: Accumulator = None) -> None:

        self.backend = backend
        self.accumulator = accumulator

        self.collection: List[Union[Collection, Tracer]] = []

    def __enter__(self) -> Collection:

        if len(self.accumulator.collector_stack) > 0:
            self.accumulator.collector_stack[-1].collection.append(self)

        self.accumulator.collector_stack.append(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(exc_val, BaseException):
            raise exc_val

        self.backend(self)

    ### BACKENDS ########

    def local_backend_execute(self) -> None:

        for executable in self.collection:

            executable.local_backend_execute()

    def accumulator_backend_handle(self, accumulator: Accumulator):

        accumulator.collector_stack.pop()

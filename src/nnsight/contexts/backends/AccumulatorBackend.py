from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from . import Backend

if TYPE_CHECKING:
    from ..accum.Accumulator import Accumulator


class AccumulatorMixin:

    def accumulator_backend_handle(self, accumulator: "Accumulator"):

        raise NotImplementedError()


class AccumulatorBackend(Backend):

    def __init__(self, accumulator: "Accumulator") -> None:

        self.accumulator = accumulator

    def __call__(self, obj: AccumulatorMixin):

        obj.accumulator_backend_handle(self.accumulator)

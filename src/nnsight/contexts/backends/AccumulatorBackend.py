from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from . import Backend

if TYPE_CHECKING:
    from ..accum.Accumulator import Accumulator


class AccumulatorMixin:
    """To be inherited by objects that want to be able to be executed by the AccumulatorBackend."""

    def accumulator_backend_handle(self, accumulator: "Accumulator") -> None:
        """Should add self to the current accumulator in some capacity.

        Args:
            accumulator (Accumulator): Current Accumulator.
        """

        raise NotImplementedError()


class AccumulatorBackend(Backend):
    """Backend to accumulate multiple context object to be executed collectively.

    Context object must inherit from AccumulatorMixin and implement its methods.

    Attributes:

        accumulator (Accumulator): Current Accumulator object.
    """

    def __init__(self, accumulator: "Accumulator") -> None:

        self.accumulator = accumulator

    def __call__(self, obj: AccumulatorMixin):

        obj.accumulator_backend_handle(self.accumulator)

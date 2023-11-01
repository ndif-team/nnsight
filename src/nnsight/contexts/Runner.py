from __future__ import annotations

from typing import TYPE_CHECKING

from .Invoker import Invoker
from .Tracer import Tracer

if TYPE_CHECKING:
    from ..models.AbstractModel import AbstractModel


class Runner(Invoker, Tracer):
    """The Runner Tracer object manages the intervention tracing for a given model's _run_local method. Also acts as an invoker.

    Example:

        A simple entering of a runner context on a language model, and running a prompt with no interventions:

        >>> with model.forward('The Eiffel Tower is in the city of') as invoker:
        >>>         pass
        >>> print(invoker.output)

    Args:
        Invoker (_type_): _description_
        Tracer (_type_): _description_
    """

    def __init__(
        self,
        model: AbstractModel,
        *args,
        inference: bool = True,
        **kwargs,
    ) -> None:
        Tracer.__init__(self, model)
        Invoker.__init__(self, self, *args, **kwargs)

        self.inference = inference

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.output = self.model(
            self.model._run_local,
            self.input,
            self.graph,
            *self.args,
            inference=self.inference,
            **self.kwargs,
        )

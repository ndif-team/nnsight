from __future__ import annotations

from typing import TYPE_CHECKING

from .Invoker import Invoker
from .Tracer import Tracer

if TYPE_CHECKING:
    from ..models.AbstractModel import AbstractModel


class Runner(Invoker, Tracer):
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

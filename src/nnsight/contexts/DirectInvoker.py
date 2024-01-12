from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from .Invoker import Invoker
from .Runner import Runner

if TYPE_CHECKING:
    from ..models.NNsightModel import NNsightModel


class DirectInvoker(Runner, Invoker):
    def __init__(
        self, model: "NNsightModel", *args, fwd_args: Dict[str, Any] = None, **kwargs
    ):
        if fwd_args is None:
            fwd_args = dict()

        self.fwd_args = fwd_args

        Runner.__init__(self, model, **self.fwd_args)

        Invoker.__init__(self, self, *args, **kwargs)

    def __enter__(self) -> DirectInvoker:

        Invoker.__enter__(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        Invoker.__exit__(self, exc_type, exc_val, exc_tb)
        self.fwd_args.pop("validate", None)
        self.kwargs = self.fwd_args
        self.args = []

        Runner.__exit__(self, exc_type, exc_val, exc_tb)

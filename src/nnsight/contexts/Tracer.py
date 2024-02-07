from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple

import torch

from ..tracing.Graph import Graph
from .Invoker import Invoker

if TYPE_CHECKING:
    from ..models.NNsightModel import NNsight


class Tracer:
    """The Tracer class creates a :class:`nnsight.tracing.Graph.Graph` around the meta_model of a :class:`nnsight.models.NNsightModel.NNsight` which tracks and manages the operations performed on the inputs and outputs of said model.

    Attributes:
        model (nnsight.models.NNsightModel.NNsight): nnsight Model object that ths context manager traces and executes.
        graph (nnsight.tracing.Graph.Graph): Graph which operations performed on the input and output of Modules are added and later executed.
        args (List[Any]): Positional arguments to be passed to function that executes the model.
        kwargs (Dict[str,Any]): Keyword arguments to be passed to function that executes the model.
        batch_size (int): Batch size of the most recent input. Used by Module to create input/output proxies.
        batch_start (int): Batch start of the most recent input. Used by Module to create input/output proxies.
        batched_input Any: Batched version of all inputs involved in this Tracer.
    """

    def __init__(
        self,
        model: "NNsight",
        validate: bool = True,
        **kwargs,
    ) -> None:

        self.model = model

        self.kwargs = kwargs

        self.graph = Graph(
            self.model.model, proxy_class=model.proxy_class, validate=validate
        )

        self.invoker: Invoker = None

        self.batch_size: int = 0
        self.batch_start: int = 0

        self.batched_input: Any = None

        # Modules need to know about the current Tracer to create the correct proxies.
        for name, module in self.model.model.named_modules():
            if not isinstance(module, torch.nn.ModuleList):
                module.tracer = self

        self.model.model.tracer = self

    def __getattr__(self, key: Any) -> Any:
        """Wrapper of meta_model's attributes to access Module's inputs and outputs.

        Returns:
            Any: Attribute.
        """
        return getattr(self.model.model, key)

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(exc_val, BaseException):
            raise exc_val

        output = self.model.interleave(
            self.model._execute,
            self.graph,
            *self.batched_input,
            **self.kwargs,
        )

    def invoke(self, *inputs: Tuple[Any], **kwargs) -> Invoker:

        if self.invoker is not None:

            raise Exception("Can't create an invoker context with one already open!")

        return Invoker(self, *inputs, **kwargs)
    
    def next(self, increment: int = 1) -> None:
        """Increments call_iter of all ``Module``s. Useful when doing iterative/generative runs.

        Args:
            increment (int): How many call_iter to increment at once. Defaults to 1.
        """

        self.model.model.reset_proxies()
        self.model.model.next()


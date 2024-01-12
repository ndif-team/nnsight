from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, List

import torch

from ..intervention import InterventionProxy
from ..tracing.Graph import Graph

if TYPE_CHECKING:
    from ..models.NNsightModel import NNsightModel


class Tracer(AbstractContextManager):
    """The Tracer class creates a :class:`nnsight.tracing.Graph.Graph` around the meta_model of a :class:`nnsight.models.NNsightModel.NNsightModel` which tracks and manages the operations performed on the inputs and outputs of said model.

    Attributes:
        model (nnsight.models.NNsightModel.NNsightModel): nnsight Model object that ths context manager traces and executes.
        graph (nnsight.tracing.Graph.Graph): Graph which operations performed on the input and output of Modules are added and later executed.
        args (List[Any]): Positional arguments to be passed to function that executes the model.
        kwargs (Dict[str,Any]): Keyword arguments to be passed to function that executes the model.
        batch_size (int): Batch size of the most recent input. Used by Module to create input/output proxies.
        batch_start (int): Batch start of the most recent input. Used by Module to create input/output proxies.
        generation_idx (int): Current generation idx for multi-iteration generation. Used by Module to create input/output proxies.
        batched_input Any: Batched version of all inputs involved in this Tracer.
        output (Any): Output of execution after __exit__
    """

    def __init__(
        self,
        model: "NNsightModel",
        *args,
        validate: bool = True,
        **kwargs,
    ) -> None:
        self.model = model

        self.args = args
        self.kwargs = kwargs

        self.graph = Graph(
            self.model.meta_model, proxy_class=InterventionProxy, validate=validate
        )

        self.batch_size: int = 0
        self.batch_start: int = 0
        self.generation_idx: int = 0
        self.batched_input: Any = None

        self.output = None

        # Modules need to know about the current Tracer to create the correct proxies.
        for name, module in self.model.meta_model.named_modules():
            if not isinstance(module, torch.nn.ModuleList):
                module.tracer = self

    @abstractmethod
    def __enter__(self) -> Tracer:
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        raise NotImplementedError()

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import torch

from ..intervention import InterventionProxy

if TYPE_CHECKING:

    from .Tracer import Tracer


class Invoker(AbstractContextManager):
    """An Invoker is meant to work in tandem with a :class:`nnsight.contexts.Tracer.Tracer` to enter input and manage intervention tracing.

    Attributes:
        tracer (nnsight.contexts.Tracer.Tracer): Tracer object to enter input and manage context.
        input (Any): Initially entered input, then post-processed input from model's _prepare_inputs method.
        scan (bool): If to use a 'meta' version of the  post-processed input to run through the model using it's _scan method,
            in order to update the potential sizes/dtypes of all module's inputs/outputs as well as validate things work correctly.
            Scanning is not free computation wise so you may want to turn this to false when running in a loop.
            When making interventions, you made get shape errors if scan is false as it validates operations based on shapes so
            for looped calls where shapes are consistent, you may want to have scan=True for the first loop. Defaults to True.
        args (List[Any]): Positional arguments passed to the model's _prepare_inputs method.
        kwargs (Dict[str,Any]): Keyword arguments passed to the model's _prepare_inputs method.
    """

    def __init__(
        self,
        tracer: "Tracer",
        *inputs: Tuple[Any],
        scan: bool = True,
        **kwargs,
    ) -> None:
        self.tracer = tracer
        self.inputs = inputs
        self.scan = scan
        self.kwargs = kwargs

    def __enter__(self) -> Invoker:
        """Enters a new invocation context with a given input.

        Calls the model's _prepare_inputs method using the input and other arguments.
        If scan is True, uses the model's _scan method to update and validate module inputs/outputs.
        Gets a batched version of the post processed input using the model's _batched_inputs method to update the Tracer's
            current batch_size and batched_input.

        Returns:
            Invoker: Invoker.
        """

        self.tracer.invoker = self

        self.inputs, batch_size = self.tracer.model._prepare_inputs(
            *self.inputs, **self.kwargs
        )

        if self.scan:
            for name, module in self.tracer.model.meta_model.named_modules():
                if not isinstance(module, torch.nn.ModuleList):
                    module.clear()
            self.tracer.model.meta_model.clear()
            self.tracer.model._scan(*self.inputs, **self.tracer.kwargs)
        else:
            for name, module in self.tracer.model.meta_model.named_modules():
                if not isinstance(module, torch.nn.ModuleList):
                    module.reset()
            self.tracer.model.meta_model.reset()

        self.tracer.batch_start += self.tracer.batch_size
        self.tracer.batch_size = batch_size

        self.tracer.batched_input = self.tracer.model._batch_inputs(
            self.tracer.batched_input, *self.inputs, 
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(exc_val, BaseException):
            raise exc_val

        self.tracer.invoker = None

    def apply(self, target: Callable, *args, **kwargs) -> InterventionProxy:
        """Helper method to directly add a function to the intervention graph.

        Args:
            target (Callable): Function to apply

        Returns:
            InterventionProxy: Proxy of applying that function.
        """
        return self.tracer.graph.add(target=target, args=args, kwargs=kwargs)

    def next(self, increment: int = 1) -> None:
        """Increments call_iter of all ``Module``s. Useful when doing iterative/generative runs.

        Args:
            increment (int): How many call_iter to increment at once. Defaults to 1.
        """

        for name, module in self.tracer.model.meta_model.named_modules():
            if not isinstance(module, torch.nn.ModuleList):
                module.reset_proxies()
                module.next(increment)

        self.tracer.model.meta_model.reset_proxies()
        self.tracer.model.meta_model.next()

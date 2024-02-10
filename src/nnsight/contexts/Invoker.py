from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

if TYPE_CHECKING:

    from .Tracer import Tracer


class Invoker(AbstractContextManager):
    """An Invoker is meant to work in tandem with a :class:`nnsight.contexts.Tracer.Tracer` to enter input and manage intervention tracing.

    Attributes:
        tracer (nnsight.contexts.Tracer.Tracer): Tracer object to enter input and manage context.
        inputs (Tuple[Any]): Initially entered inputs, then post-processed inputs from model's ._prepare_inputs(...) method.
        scan (bool): If to execute the model using `FakeTensor` in order to update the potential sizes/dtypes of all modules' Envoys' inputs/outputs as well as validate things work correctly.
            Scanning is not free computation wise so you may want to turn this to false when running in a loop.
            When making interventions, you made get shape errors if scan is false as it validates operations based on shapes so
            for looped calls where shapes are consistent, you may want to have scan=True for the first loop. Defaults to True.
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
        If scan is True, uses the model's ._execute method to update and validate module Envoy's inputs/outputs using a fake mode.
        Gets a batched version of the post processed input using the model's ._batched_inputs method to update the Tracer's
            current batch_size and batched_input.

        Returns:
            Invoker: Invoker.
        """

        self.tracer._invoker = self

        self.inputs, batch_size = self.tracer._model._prepare_inputs(
            *self.inputs, **self.kwargs
        )

        if self.scan:
            self.tracer._model._envoy._clear()

            with FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(assume_static_by_default=True),
            ) as fake_mode:
                with FakeCopyMode(fake_mode):
                    self.tracer._model._execute(
                        *copy.deepcopy(self.inputs),
                        **copy.deepcopy(self.tracer._kwargs),
                    )
        else:
            self.tracer._model._envoy._reset()

        self.tracer._batch_start += self.tracer._batch_size
        self.tracer._batch_size = batch_size

        self.tracer._batched_input = self.tracer._model._batch_inputs(
            self.tracer._batched_input,
            *self.inputs,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(exc_val, BaseException):
            raise exc_val

        self.tracer._invoker = None

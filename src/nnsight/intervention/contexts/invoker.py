from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ... import util
from ...tracing.contexts.globals import GlobalTracingContext
from ..graph import InterventionNode, InterventionProxy, InterventionProxyType

if TYPE_CHECKING:

    from . import InterleavingTracer


class Invoker(AbstractContextManager):
    """An Invoker is meant to work in tandem with a :class:`nnsight.intervention.contexts.InterleavingTracer` to enter input and manage intervention tracing.

    Attributes:
        tracer (nnsight.contexts.Tracer.Tracer): Tracer object to enter input and manage context.
        inputs (tuple[Any]): Initially entered inputs, then post-processed inputs from model's ._prepare_inputs(...) method.
        scan (bool): If to execute the model using `FakeTensor` in order to update the potential sizes/dtypes of all modules' Envoys' inputs/outputs as well as validate things work correctly.
            Scanning is not free computation wise so you may want to turn this to false when running in a loop.
            When making interventions, you made get shape errors if scan is false as it validates operations based on shapes so
            for looped calls where shapes are consistent, you may want to have scan=True for the first loop. Defaults to False.
        kwargs (Dict[str,Any]): Keyword arguments passed to the model's _prepare_inputs method.
        scanning (bool): If currently scanning.
    """

    def __init__(
        self,
        tracer: "InterleavingTracer",
        *args,
        scan: bool = False,
        **kwargs,
    ) -> None:

        self.tracer = tracer
        self.inputs = (args, kwargs)

        self.scan = scan

        self.scanning = False

        self.tracer.invoker = self

        self.batch_size: Optional[int] = None

    def __enter__(self) -> Invoker:
        """Enters a new invocation context with a given input.

        Calls the model's _prepare_inputs method using the input and other arguments.
        If scan is True, uses the model's ._execute method to update and validate module Envoy's inputs/outputs using a fake mode.
        Gets a batched version of the post processed input using the model's ._batched_inputs method to update the Tracer's
            current batch_size and batched_input.

        Returns:
            Invoker: Invoker.
        """

        has_proxies_in_inputs = False
        
        def check_for_proxies(proxy: InterventionProxyType):

            nonlocal has_proxies_in_inputs

            has_proxies_in_inputs = True

            return proxy

        # We need to check if there were any Proxies in the actual Invoker input. This might be True in a Session where values from one trace are used as an input to another.
        util.apply(self.inputs, check_for_proxies, InterventionProxy)

        # We dont want to create new proxies during scanning/prepare_inputs so we exit the global tracing context.
        with GlobalTracingContext.exit_global_tracing_context():

            # If we dont have proxies we can immediately prepare the input so the user can see it and the batch_size.
            if not has_proxies_in_inputs:

                self.inputs, self.batch_size = self.tracer._model._prepare_input(
                    *self.inputs[0], **self.inputs[1]
                )

            if self.scan:

                input = self.inputs

                if has_proxies_in_inputs:

                    input = util.apply(input, lambda x: x.fake_value, InterventionNode)

                    input, _ = self.tracer._model._prepare_input(*input[0], **input[1])

                # Clear all fake inputs and outputs because were going to re-populate them.
                self.tracer._model._envoy._clear()

                self.scanning = True

                with util.Patcher() as patcher:

                    # Some logic (like gpt-j rotary embeddings) gets "poisoned" by FakeTensors.
                    # This does not happen when `torch._jit_internal.is_scripting() returns True.`
                    patcher.add(
                        util.Patch(torch._jit_internal, lambda: True, "is_scripting")
                    )

                    with FakeTensorMode(
                        allow_non_fake_inputs=True,
                        shape_env=ShapeEnv(assume_static_by_default=True),
                    ) as fake_mode:
                        with FakeCopyMode(fake_mode):
                            fn = (
                                self.tracer._model._execute
                                if self.tracer.args[0] is None
                                else getattr(self.tracer._model, self.tracer.args[0])
                            )
                            fn(
                                *copy.deepcopy(input[0]),
                                **copy.deepcopy(input[1]),
                                **copy.deepcopy(self.tracer.kwargs),
                            )

                self.scanning = False

            else:
                self.tracer._model._envoy._reset()

            self.tracer.args.append(self.inputs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.tracer.invoker = None

        if isinstance(exc_val, BaseException):
            raise exc_val

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

    from . import InterventionTracer


class Invoker(AbstractContextManager):
    """An Invoker is meant to work in tandem with a :class:`nnsight.contexts.Tracer.Tracer` to enter input and manage intervention tracing.

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
        tracer: "InterventionTracer",
        *args,
        scan: bool = False,
        **kwargs,
    ) -> None:

        self.tracer = tracer
        self.input = (args, kwargs)

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

        self.input = util.apply(self.input, check_for_proxies, InterventionProxy)

        with GlobalTracingContext.exit_global_tracing_context():

            if not has_proxies_in_inputs:

                self.input, self.batch_size = self.tracer._model._prepare_input(
                    *self.input[0], **self.input[1]
                )

            if self.scan:

                input = self.input

                if has_proxies_in_inputs:

                    input = util.apply(input, lambda x: x.fake_value, InterventionNode)

                    input, _ = self.tracer.session.model._prepare_input(*input[0], **input[1])

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

            self.tracer.args.append(self.input)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.tracer.invoker = None

        if isinstance(exc_val, BaseException):
            raise exc_val

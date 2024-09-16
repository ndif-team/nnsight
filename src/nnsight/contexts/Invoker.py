from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .. import util
from ..patching import Patch, Patcher
from ..tracing.Node import Node
from ..tracing.Proxy import Proxy
from . import check_for_dependencies
from .GraphBasedContext import GlobalTracingContext

if TYPE_CHECKING:

    from .Tracer import Tracer


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
        tracer: "Tracer",
        *inputs: Any,
        scan: bool = False,
        **kwargs,
    ) -> None:

        self.tracer = tracer
        self.inputs = inputs
        self.scan = scan
        self.kwargs = kwargs

        self.scanning = False

        self.tracer.invoker = self

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

        # If were accumulating, we might have Proxies in the input.
        # Therefore we first: Check to see if there are any Proxies.
        # If there are, preserve the raw inputs with Proxies converted to a Locked Bridge protocol.
        # Set self.inputs to be the proxy_value so we can prepare_inputs, get the batch size, and scan.
        if self.tracer.model._session is not None:

            self.inputs, has_proxies_in_inputs = check_for_dependencies(
                self.inputs
            )

        with GlobalTracingContext.exit_global_tracing_context():

            if not has_proxies_in_inputs:

                self.inputs, batch_size = self.tracer.model._prepare_inputs(
                    *self.inputs, **self.kwargs
                )

            if self.scan:

                inputs = self.inputs

                if has_proxies_in_inputs:

                    inputs = util.apply(inputs, lambda x: x.proxy_value, Node)

                    inputs, batch_size = self.tracer.model._prepare_inputs(
                        *inputs, **self.kwargs
                    )

                self.tracer.model._envoy._clear()

                self.scanning = True

                with Patcher() as patcher:

                    # Some logic (like gpt-j rotary embeddings) gets "poisoned" by FakeTensors.
                    # This does not happen when `torch._jit_internal.is_scripting() returns True.`
                    patcher.add(
                        Patch(torch._jit_internal, lambda: True, "is_scripting")
                    )

                    with FakeTensorMode(
                        allow_non_fake_inputs=True,
                        shape_env=ShapeEnv(assume_static_by_default=True),
                    ) as fake_mode:
                        with FakeCopyMode(fake_mode):
                            self.tracer.model._execute(
                                *copy.deepcopy(inputs),
                                **copy.deepcopy(self.tracer._kwargs),
                            )

                self.scanning = False

            else:
                self.tracer.model._envoy._reset()

            self.tracer._invoker_inputs.append(self.inputs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.tracer.invoker = None

        if isinstance(exc_val, BaseException):
            raise exc_val

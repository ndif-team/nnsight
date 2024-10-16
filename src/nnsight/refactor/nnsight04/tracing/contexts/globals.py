from __future__ import annotations

import inspect
from contextlib import AbstractContextManager
from functools import wraps
from inspect import getmembers, isclass
from typing import Any

import torch
from torch.overrides import TorchFunctionMode
from torch.utils import data

from ... import util
from ..graph import Graph
from . import Context


def global_patch(root, name: str) -> util.Patch:

    fn = getattr(root, name)

    @wraps(fn)
    def inner(*args, **kwargs):

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(fn, *args, **kwargs)

    return util.Patch(root, inner, name)


def global_patch_class(cls: type) -> util.Patch:

    if cls.__new__ is object.__new__:

        def super_new(cls, *args, **kwargs):

            return object.__new__(cls)

        cls.__new__ = super_new

    fn = cls.__new__

    @wraps(fn)
    def inner(cls, *args, **kwargs):

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(cls, *args, **kwargs)

    return util.Patch(cls, inner, "__new__")


class GlobalTracingContext(Context):
    """The Global Tracing Context handles adding tracing operations globally without reference to a given `GraphBasedContext`.
    There should only be one of these and that is `GlobalTracingContext.GLOBAL_TRACING_CONTEXT`.
    `GlobalTracingContext.TORCH_HANDLER` handles adding torch functions without reference to a given `GraphBasedContext`.

    """

    GLOBAL_TRACING_CONTEXT: GlobalTracingContext
    TORCH_HANDLER: GlobalTracingContext.GlobalTracingTorchHandler
    PATCHER: util.Patcher = util.Patcher(
        [
            global_patch_class(torch.nn.Parameter),
            global_patch_class(data.DataLoader),
            global_patch(torch, "arange"),
            global_patch(torch, "empty"),
            global_patch(torch, "eye"),
            global_patch(torch, "full"),
            global_patch(torch, "linspace"),
            global_patch(torch, "logspace"),
            global_patch(torch, "ones"),
            global_patch(torch, "rand"),
            global_patch(torch, "randint"),
            global_patch(torch, "randn"),
            global_patch(torch, "randperm"),
            global_patch(torch, "zeros"),
            global_patch(torch, "cat"),
        ]
        + [
            global_patch_class(value)
            for key, value in getmembers(torch.optim, isclass)
            if issubclass(value, torch.optim.Optimizer)
        ]
    )

    class GlobalTracingTorchHandler(TorchFunctionMode):

        def __torch_function__(self, func, types, args, kwargs=None):

            if kwargs is None:

                kwargs = {}

            if "_VariableFunctionsClass" in func.__qualname__:
                return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(
                    func, *args, **kwargs
                )

            return func(*args, **kwargs)

    class GlobalTracingExit(AbstractContextManager):

        def __enter__(self) -> Any:

            GlobalTracingContext.TORCH_HANDLER.__exit__(None, None, None)
            GlobalTracingContext.PATCHER.__exit__(None, None, None)

            return self

        def __exit__(self, exc_type, exc_val, traceback):

            GlobalTracingContext.TORCH_HANDLER.__enter__()
            GlobalTracingContext.PATCHER.__enter__()

            if isinstance(exc_val, BaseException):

                raise exc_val

    def __init__(self) -> None:
        """We create an empty `GraphBasedContext` by default."""

        self.graph: Graph = None

    @staticmethod
    def exit_global_tracing_context():

        return GlobalTracingContext.GlobalTracingExit()

    @staticmethod
    def try_register(graph_based_context: Context) -> bool:
        """Attempts to register a `Graph` globally.]
        Will not if one is already registered.

        Args:
            graph_based_context (GraphBasedContext): `GraphBasedContext` to register.

        Returns:
            bool: True if registering ws successful, False otherwise.
        """

        if GlobalTracingContext.GLOBAL_TRACING_CONTEXT:

            return False

        GlobalTracingContext.register(graph_based_context)

        return True

    @staticmethod
    def try_deregister(graph_based_context: Context) -> bool:
        """Attempts to deregister a `Graph` globally.
        Will not if `graph_based_context` does not have the same `Graph` as the currently registered one.

        Args:
            graph_based_context (GraphBasedContext): `GraphBasedContext` to deregister.

        Returns:
            bool: True if deregistering ws successful, False otherwise.
        """
        if (
            not GlobalTracingContext.GLOBAL_TRACING_CONTEXT
            or graph_based_context.graph
            is not GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph
        ):

            return False
        GlobalTracingContext.deregister()

        return True

    @staticmethod
    def register(graph_based_context: Context) -> None:
        """Register `GraphBasedContext` globally.

        Args:
            graph_based_context (GraphBasedContext): GraphBasedContext to register.
        """

        assert GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is None

        GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph = graph_based_context.graph

        GlobalTracingContext.TORCH_HANDLER.__enter__()
        GlobalTracingContext.PATCHER.__enter__()

    @staticmethod
    def deregister() -> None:
        """Deregister `GraphBasedContext` globally.

        Args:
            graph_based_context (GraphBasedContext): GraphBasedContext to deregister.
        """

        assert GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is not None

        GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph = None

        GlobalTracingContext.TORCH_HANDLER.__exit__(None, None, None)
        GlobalTracingContext.PATCHER.__exit__(None, None, None)

    def __bool__(self) -> bool:
        """True if there is a `GraphBasedContext` registered globally. False otherwise."""

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is not None

    def __getattribute__(self, name: str) -> Any:
        """Prevent attribute access if no `GraphBasedContext` registered."""

        static_methods = [
            name
            for name, value in inspect.getmembers(Context, predicate=inspect.ismethod)
        ]

        if name in static_methods:

            if not GlobalTracingContext.GLOBAL_TRACING_CONTEXT:

                raise Exception(
                    "Global ops cannot be used outside of a tracing context."
                )

        return object.__getattribute__(self, name)


GlobalTracingContext.GLOBAL_TRACING_CONTEXT = GlobalTracingContext()
GlobalTracingContext.TORCH_HANDLER = GlobalTracingContext.GlobalTracingTorchHandler()

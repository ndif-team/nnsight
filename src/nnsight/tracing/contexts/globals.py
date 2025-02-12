from __future__ import annotations

import inspect
from contextlib import AbstractContextManager
from functools import wraps
from types import FunctionType, MethodType
from typing import Any, Type, Union

from ... import util, CONFIG
from ..graph import Graph
from . import Tracer


def global_patch_class(cls: type) -> util.Patch:

    if cls.__new__ is object.__new__:

        def super_new(cls, *args, **kwargs):

            return object.__new__(cls)

        cls.__new__ = super_new

    fn = cls.__new__

    @wraps(fn)
    def inner(cls, *args, **kwargs):

        if not GlobalTracingContext.GLOBAL_TRACING_CONTEXT:
            return cls(*args, **kwargs)

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(cls, *args, **kwargs)

    return util.Patch(cls, inner, "__new__")


def global_patch_fn(fn: FunctionType) -> util.Patch:

    @wraps(fn)
    def inner(*args, **kwargs):

        if not GlobalTracingContext.GLOBAL_TRACING_CONTEXT:
            return fn(*args, **kwargs)

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(fn, *args, **kwargs)

    return util.Patch(inspect.getmodule(fn), inner, fn.__name__)


def global_patch_method(cls: type, fn: MethodType) -> None:

    @wraps(fn)
    def inner(*args, **kwargs):

        if not GlobalTracingContext.GLOBAL_TRACING_CONTEXT:
            return fn(*args, **kwargs)

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.apply(fn, *args, **kwargs)

    patch = util.Patch(cls, inner, fn.__name__)

    GlobalTracingContext.PATCHER.add(patch)


def global_patch(obj: Union[FunctionType, Type]):

    if isinstance(obj, type):

        patch = global_patch_class(obj)

    else:

        patch = global_patch_fn(obj)

    GlobalTracingContext.PATCHER.add(patch)


class GlobalTracingContext(Tracer):
    """The Global Tracing Context handles adding tracing operations globally without reference to a given `GraphBasedContext`.
    There should only be one of these and that is `GlobalTracingContext.GLOBAL_TRACING_CONTEXT`.
    `GlobalTracingContext.TORCH_HANDLER` handles adding torch functions without reference to a given `GraphBasedContext`.

    """

    GLOBAL_TRACING_CONTEXT: GlobalTracingContext
    PATCHER: util.Patcher = util.Patcher()

    class GlobalTracingExit(AbstractContextManager):

        def __enter__(self) -> Any:

            if CONFIG.APP.GLOBAL_TRACING:

                GlobalTracingContext.PATCHER.__exit__(None, None, None)

            return self

        def __exit__(self, exc_type, exc_val, traceback):

            if CONFIG.APP.GLOBAL_TRACING:

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
    def try_register(graph_based_context: Tracer) -> bool:
        """Attempts to register a `Graph` globally.]
        Will not if one is already registered.

        Args:
            graph_based_context (GraphBasedContext): `GraphBasedContext` to register.

        Returns:
            bool: True if registering ws successful, False otherwise.
        """

        if CONFIG.APP.GLOBAL_TRACING:

            if GlobalTracingContext.GLOBAL_TRACING_CONTEXT:

                return False

            GlobalTracingContext.register(graph_based_context)

        return True

    @staticmethod
    def try_deregister(graph_based_context: Tracer) -> bool:
        """Attempts to deregister a `Graph` globally.
        Will not if `graph_based_context` does not have the same `Graph` as the currently registered one.

        Args:
            graph_based_context (GraphBasedContext): `GraphBasedContext` to deregister.

        Returns:
            bool: True if deregistering ws successful, False otherwise.
        """

        if CONFIG.APP.GLOBAL_TRACING:

            if (
                not GlobalTracingContext.GLOBAL_TRACING_CONTEXT
                or graph_based_context.graph
                is not GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph
            ):

                return False

            GlobalTracingContext.deregister()

        return True

    @staticmethod
    def register(graph_based_context: Tracer) -> None:
        """Register `GraphBasedContext` globally.

        Args:
            graph_based_context (GraphBasedContext): GraphBasedContext to register.
        """

        assert GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is None

        GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph = graph_based_context.graph

        GlobalTracingContext.PATCHER.__enter__()

    @staticmethod
    def deregister() -> None:
        """Deregister `GraphBasedContext` globally.

        Args:
            graph_based_context (GraphBasedContext): GraphBasedContext to deregister.
        """

        assert GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is not None

        GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph = None

        GlobalTracingContext.PATCHER.__exit__(None, None, None)

    def __bool__(self) -> bool:
        """True if there is a `GraphBasedContext` registered globally. False otherwise."""

        return GlobalTracingContext.GLOBAL_TRACING_CONTEXT.graph is not None


GlobalTracingContext.GLOBAL_TRACING_CONTEXT = GlobalTracingContext()

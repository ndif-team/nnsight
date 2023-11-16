from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, Union

import torch

from .. import util

if TYPE_CHECKING:
    from .Node import Node


class Proxy:
    """Proxy objects are the actual objects that interact with operations in order to update the graph to create new nodes.

    The operations that are traceable on base Proxy objects are many python built-in and magic methods, as well as implementing __torch_function__ to trace torch operations.

    Attributes:
        node (Node): This proxy's node.
    """

    @staticmethod
    def proxy_update(value1: Any, value2: Any) -> None:
        """Updates Tensor values with other Tensor values.

        Args:
            value1 (Any): Collection with Tensors to update.
            value2 (Any): Collection with Tensors to pull values from.
        """
        if isinstance(value1, torch.Tensor):
            value1[:] = value2
        elif isinstance(value1, list) or isinstance(value1, tuple):
            for value_idx in range(len(value1)):
                Proxy.proxy_update(value1[value_idx], value2[value_idx])
        elif isinstance(value1, dict):
            for key in value1:
                Proxy.proxy_update(value1[key], value2[key])

    @staticmethod
    def proxy_call(callable: Callable, *args, **kwargs) -> None:
        return callable(*args, **kwargs)

    def __init__(self, node: "Node") -> None:
        self.node = node

    def __call__(self, *args, **kwargs) -> Proxy:
        """
        Calling a Proxy object normally just creates a Proxy.proxy_call operation. However if this call is a method on the root module proxy, it's assumed that one wishes to trace into the method and therefore trace all operations inside it.

        Returns:
            Proxy: New call proxy.
        """
        # If calling a method (not a sub-module) on the main module of this graph,
        # we want to trace into that method.
        if self.node.args[0] is self.node.graph.module_proxy.node and not isinstance(
            self.node.proxy_value, torch.nn.Module
        ):
            value = self.node.proxy_value.__func__(
                self.node.graph.module_proxy, *args, **kwargs
            )

            return value
        # Otherwise we just want to add a node saying we wish to call this module.
        else:
            return self.node.graph.add(
                target=Proxy.proxy_call,
                args=[self.node] + list(args),
                kwargs=kwargs,
            )

    def __getitem__(self, key: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.getitem,
            args=[self.node, key],
        )

    def __setitem__(self, key: Union[Proxy, Any], value: Union[Proxy, Any]) -> None:
        item_proxy = self[key]

        item_proxy.node.graph.add(
            target=Proxy.proxy_update,
            args=[item_proxy.node, value],
        )

    def __getattr__(self, key: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=util.fetch_attr,
            args=[self.node, key],
        )

    def __len__(self) -> Proxy:
        return self.node.graph.add(
            target=len,
            args=[self.node],
        )

    def __add__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.add,
            args=[self.node, other],
        )

    def __sub__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.sub,
            args=[self.node, other],
        )

    def __pow__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=pow,
            args=[self.node, other],
        )

    def __mul__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.mul,
            args=[self.node, other],
        )

    def __matmul__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.matmul,
            args=[self.node, other],
        )

    def __truediv__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.truediv,
            args=[self.node, other],
        )

    def __bool__(self) -> bool:
        return self.node.proxy_value.__bool__()

    def __index__(self) -> int:
        return self.node.proxy_value.__index__()

    def __instancecheck__(self, __instance: Any) -> bool:
        return self.node.proxy_value.__instancecheck__(__instance)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None) -> Proxy:
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()

        proxy: Proxy = None

        for arg in args:
            if isinstance(arg, Proxy):
                proxy = arg
                break

        return proxy.node.graph.add(
            target=orig_method,
            args=args,
            kwargs=kwargs,
        )


from functools import wraps


def proxy_wrapper(fn) -> None:
    """Wraps problematic functions (torch functions sometimes).
    Checks if any of its args are proxies. If so we return a proxy of the function.
    Otherwise just run the function.

    Args:
        fn (function): Function to wrap.

    Returns:
        function: Wrapped function.
    """

    @wraps(fn)
    def patched(*args, **kwargs):
        arguments = list(args) + list(kwargs.values())

        node = None

        for arg in arguments:
            if isinstance(arg, Proxy):
                node = arg.node

                break

        if node is not None:
            value = fn(
                *node.prepare_proxy_values(args),
                **node.prepare_proxy_values(kwargs),
            )

            return node.graph.add(value=value, target=fn, args=args, kwargs=kwargs)

        else:
            return fn(*args, **kwargs)

    return patched

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
    def proxy_call(callable: Callable, *args, **kwargs) -> None:
        return callable(*args, **kwargs)

    def __init__(self, node: "Node") -> None:
        self.node = node

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d: dict):
        self.__dict__ = d

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
        self.node.graph.add(
            target=operator.setitem,
            args=[self.node, key, value],
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

    def __abs__(self) -> Proxy:
        return self.node.graph.add(
            target=operator.abs,
            args=[self.node],
        )

    def __invert__(self) -> Proxy:
        return self.node.graph.add(
            target=operator.invert,
            args=[self.node],
        )

    def __add__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.add,
            args=[self.node, other],
        )

    def __radd__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.add,
            args=[other, self.node],
        )

    def __sub__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.sub,
            args=[self.node, other],
        )

    def __rsub__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.sub,
            args=[other, self.node],
        )

    def __pow__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.pow,
            args=[self.node, other],
        )

    def __rpow__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.pow,
            args=[other, self.node],
        )

    def __mul__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.mul,
            args=[self.node, other],
        )

    def __rmul__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.mul,
            args=[other, self.node],
        )

    def __mod__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.mod,
            args=[self.node, other],
        )

    def __rmod__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.mod,
            args=[other, self.node],
        )

    def __matmul__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.matmul,
            args=[self.node, other],
        )

    def __rmatmul__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.matmul,
            args=[other, self.node],
        )

    def __truediv__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.truediv,
            args=[self.node, other],
        )

    def __rtruediv__(self, other: Union[Proxy, Any]) -> Proxy:
        return self.node.graph.add(
            target=operator.truediv,
            args=[other, self.node],
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

        def get_proxy(arg):
            nonlocal proxy

            proxy = arg

        util.apply(args, get_proxy, Proxy)

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

        def get_node(proxy: Proxy):
            nonlocal node

            node = proxy.node

        util.apply(list(args) + list(kwargs.values()), get_node, Proxy)

        if node is not None:
            return node.graph.add(target=fn, args=args, kwargs=kwargs)

        else:
            return fn(*args, **kwargs)

    return patched

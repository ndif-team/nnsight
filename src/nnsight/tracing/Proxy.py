from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, Union

import torch
from typing_extensions import Self

from .. import util

if TYPE_CHECKING:
    from .Node import Node


class Proxy:
    """Proxy objects are the actual objects that interact with operations in order to update the graph to create new nodes.

    The operations that are traceable on base Proxy objects are many python built-in and magic methods, as well as implementing __torch_function__ to trace torch operations.

    Attributes:
        node (Node): This proxy's node.
    """

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d: dict):
        self.__dict__ = d

    @staticmethod
    def proxy_call(callable: Callable, *args, **kwargs) -> Self:
        return callable(*args, **kwargs)

    def __init__(self, node: "Node") -> None:

        self.__dict__["node"] = node

        self.node: "Node"

    @property
    def value(self) -> Any:
        """Property to return the value of this proxy's node.

        Returns:
            Any: The stored value of the proxy, populated during execution of the model.
        """

        if not self.node.done():
            raise ValueError("Accessing Proxy value before it's been set.")

        return self.node.value

    def __str__(self) -> str:

        if not self.node.graph.tracing:

            return str(self.value)

        return (
            f"{type(self).__name__} ({self.node.name}): {self.node.proxy_value or ''}"
        )

    def __repr__(self) -> str:

        if not self.node.graph.tracing:

            return repr(self.value)

        return str(self)

    def __call__(self, *args, **kwargs) -> Self:
        """
        Calling a Proxy object just creates a Proxy.proxy_call operation.

        Returns:
            Proxy: New call proxy.
        """

        return self.node.graph.add(
            target=Proxy.proxy_call,
            args=[self.node] + list(args),
            kwargs=kwargs,
        )

    def __getitem__(self, key: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.getitem,
            args=[self.node, key],
        )

    def __setitem__(self, key: Union[Proxy, Any], value: Union[Self, Any]) -> None:
        self.node.graph.add(
            target=operator.setitem,
            args=[self.node, key, value],
        )

    def __getattr__(self, key: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=util.fetch_attr,
            args=[self.node, key],
        )

    def __setattr__(self, key: Union[Proxy, Any], value: Union[Self, Any]) -> None:

        if key == "__dict__":

            super().__setattr__(key, value)

            return

        return self.node.graph.add(
            target=setattr,
            args=[self.node, key, value],
        )

    def __len__(self) -> Self:
        return self.node.graph.add(
            target=len,
            args=[self.node],
        )

    def __abs__(self) -> Self:
        return self.node.graph.add(
            target=operator.abs,
            args=[self.node],
        )

    def __invert__(self) -> Self:
        return self.node.graph.add(
            target=operator.invert,
            args=[self.node],
        )

    def __add__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.add,
            args=[self.node, other],
        )

    def __radd__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.add,
            args=[other, self.node],
        )

    def __sub__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.sub,
            args=[self.node, other],
        )

    def __rsub__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.sub,
            args=[other, self.node],
        )

    def __pow__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.pow,
            args=[self.node, other],
        )

    def __rpow__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.pow,
            args=[other, self.node],
        )

    def __mul__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.mul,
            args=[self.node, other],
        )

    def __rmul__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.mul,
            args=[other, self.node],
        )

    def __mod__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.mod,
            args=[self.node, other],
        )

    def __rmod__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.mod,
            args=[other, self.node],
        )

    def __matmul__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.matmul,
            args=[self.node, other],
        )

    def __rmatmul__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.matmul,
            args=[other, self.node],
        )

    def __truediv__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.graph.add(
            target=operator.truediv,
            args=[self.node, other],
        )

    def __rtruediv__(self, other: Union[Proxy, Any]) -> Self:
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
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None) -> Self:
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

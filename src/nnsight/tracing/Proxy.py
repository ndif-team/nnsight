from __future__ import annotations

import inspect
import operator
import weakref
from typing import TYPE_CHECKING, Any, Callable, Union

from typing_extensions import Self

from .. import util

if TYPE_CHECKING:
    from .Node import Node


class Proxy:
    """Proxy objects are the actual objects that interact with operations in order to update the graph to create new Nodes.

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
        self.node.proxy = weakref.proxy(self)

    @property
    def value(self) -> Any:
        """Property to return the value of this proxy's node.

        Returns:
            Any: The stored value of the proxy, populated during execution of the model.
        """

        return self.node.value

    def __str__(self) -> str:

        if not self.node.attached():

            return str(self.value)

        return f"{type(self).__name__} ({self.node.name}): {self.node.proxy_value if self.node.proxy_value is not inspect._empty else ''}"

    def __repr__(self) -> str:

        if not self.node.attached():

            return repr(self.value)

        return str(self)

    def __call__(self, *args, **kwargs) -> Self:
        """
        Calling a Proxy object just creates a Proxy.proxy_call operation.

        Returns:
            Proxy: New call proxy.
        """

        return self.node.create(
            target=Proxy.proxy_call,
            args=[self.node] + list(args),
            kwargs=kwargs,
        )

    def __getitem__(self, key: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.getitem,
            args=[self.node, key],
        )

    def __setitem__(self, key: Union[Proxy, Any], value: Union[Self, Any]) -> None:
        self.node.create(
            target=operator.setitem,
            args=[self.node, key, value],
        )

    def __getattr__(self, key: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=util.fetch_attr,
            args=[self.node, key],
        )

    def __setattr__(self, key: Union[Proxy, Any], value: Union[Self, Any]) -> None:

        if key == "__dict__":

            super().__setattr__(key, value)

            return

        return self.node.create(
            target=setattr,
            args=[self.node, key, value],
        )

    def __len__(self) -> Self:
        return self.node.create(
            target=len,
            args=[self.node],
        )

    def __abs__(self) -> Self:
        return self.node.create(
            target=operator.abs,
            args=[self.node],
        )

    def __invert__(self) -> Self:
        return self.node.create(
            target=operator.invert,
            args=[self.node],
        )

    def __neg__(self) -> Self:
        return self.node.create(
            target=operator.neg,
            args=[self.node],
        )

    def __index__(self) -> Self:
        return self.node.create(target=operator.index, args=[self.node])

    def __add__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.add,
            args=[self.node, other],
        )

    def __radd__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.add,
            args=[other, self.node],
        )

    def __sub__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.sub,
            args=[self.node, other],
        )

    def __rsub__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.sub,
            args=[other, self.node],
        )

    def __pow__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.pow,
            args=[self.node, other],
        )

    def __rpow__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.pow,
            args=[other, self.node],
        )

    def __mul__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.mul,
            args=[self.node, other],
        )

    def __rmul__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.mul,
            args=[other, self.node],
        )

    def __mod__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.mod,
            args=[self.node, other],
        )

    def __rmod__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.mod,
            args=[other, self.node],
        )

    def __matmul__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.matmul,
            args=[self.node, other],
        )

    def __rmatmul__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.matmul,
            args=[other, self.node],
        )

    def __truediv__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.truediv,
            args=[self.node, other],
        )

    def __rtruediv__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(
            target=operator.truediv,
            args=[other, self.node],
        )

    def __floordiv__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.add(
            target=operator.floordiv,
            args=[self.node, other],
        )

    def __rfloordiv__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.add(
            target=operator.floordiv,
            args=[other, self.node],
        )

    def __eq__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(target=operator.eq, args=[self.node, other])

    def __ne__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(target=operator.ne, args=[self.node, other])

    def __lt__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(target=operator.lt, args=[self.node, other])

    def __gt__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(target=operator.gt, args=[self.node, other])

    def __le__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(target=operator.le, args=[self.node, other])

    def __ge__(self, other: Union[Proxy, Any]) -> Self:
        return self.node.create(target=operator.ge, args=[self.node, other])

    def __index__(self) -> Self:
        return self.node.create(target=operator.index, args=[self.node])

    def __bool__(self) -> bool:
        return self.node.proxy_value.__bool__()

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

        return proxy.node.create(
            target=orig_method,
            args=args,
            kwargs=kwargs,
        )


from functools import wraps


def proxy_wrapper(fn) -> Callable:
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

        node = None

        def get_node(proxy: Proxy):
            nonlocal node

            node = proxy.node

        util.apply((args, kwargs), get_node, Proxy)

        if node is not None:
            return node.create(target=fn, args=args, kwargs=kwargs)

        else:
            return fn(*args, **kwargs)

    return patched

from __future__ import annotations

import inspect
import operator
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar, Union

from typing_extensions import Self

from ... import CONFIG, util
from .. import protocols

if TYPE_CHECKING:
    from .node import Node


class Proxy:
    """Proxy objects are the actual objects that interact with operations in order to update the graph to create new Nodes.

    The operations that are traceable on base Proxy objects are many python built-in and magic methods.

    Attributes:
        node (NodeType): This proxy's Node.
    """

    def __init__(self, node: "Node") -> None:

        self.__dict__["node"] = node

        self.node: "Node"

    ### API ##############################

    def save(self) -> Self:
        """Adds a lock Node to prevent its value from being cleared where normally it would be cleared when its no longer needed to save memory.
        Used to access values outside of the tracing context, after execution.

        Returns:
            InterventionProxy: Proxy.
        """

        # Add a 'lock' node with the save proxy as an argument to ensure the values are never deleted.
        # This is because 'lock' nodes never actually get set and therefore there will always be a
        # dependency for the save proxy.

        protocols.LockProtocol.add(self.node)

        return self

    def stop(self) -> None:
        protocols.StopProtocol.add(
            self.node.graph,
            self.node,
        )

    @property
    def value(self) -> Any:
        """Property to return the value of this proxy's node.

        Returns:
            Any: The stored value of the proxy, populated during execution of the model.
        """

        return self.node.value

    def __str__(self) -> str:

        if not self.node.attached:

            return str(self.value)

        return f"{type(self).__name__} ({self.node.target.__name__})"

    def __repr__(self) -> str:

        if not self.node.attached:

            return repr(self.value)

        return str(self)

    ### Special ################

    @staticmethod
    def call(callable: Callable, *args, **kwargs) -> Self:
        return callable(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Self:
        """
        Calling a Proxy object just creates a Proxy.proxy_call operation.

        Returns:
            Proxy: New call proxy.
        """

        return self.node.create(
            Proxy.call,
            *([self.node] + list(args)),
            **kwargs,
        )

    def __getattr__(self, key: Union[Self, Any]) -> Self:
        return self.node.create(util.fetch_attr, self.node, key)

    def __setattr__(self, key: Union[Proxy, Any], value: Union[Self, Any]) -> None:

        if key == "__dict__":

            super().__setattr__(key, value)

            return

        return self.node.create(
            setattr,
            self.node,
            key,
            value,
        )

    ### Regular Operators #########################

    def __getitem__(self, key: Union[Self, Any]) -> Self:
        return self.node.create(operator.getitem, self.node, key)

    def __setitem__(self, key: Union[Self, Any], value: Union[Self, Any]) -> None:
        self.node.create(
            operator.setitem,
            self.node,
            key,
            value,
        )

    def __abs__(self) -> Self:
        return self.node.create(
            operator.abs,
            self.node,
        )

    def __invert__(self) -> Self:
        return self.node.create(
            operator.invert,
            self.node,
        )

    def __neg__(self) -> Self:
        return self.node.create(
            operator.neg,
            self.node,
        )

    def __add__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(operator.add, self.node, other)

    def __radd__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.add,
            other,
            self.node,
        )

    def __sub__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.sub,
            self.node,
            other,
        )

    def __rsub__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(operator.sub, other, self.node)

    def __pow__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.pow,
            self.node,
            other,
        )

    def __rpow__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.pow,
            other,
            self.node,
        )

    def __mul__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.mul,
            self.node,
            other,
        )

    def __rmul__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.mul,
            other,
            self.node,
        )

    def __mod__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.mod,
            self.node,
            other,
        )

    def __rmod__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.mod,
            other,
            self.node,
        )

    def __matmul__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.matmul,
            self.node,
            other,
        )

    def __rmatmul__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.matmul,
            other,
            self.node,
        )

    def __truediv__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.truediv,
            self.node,
            other,
        )

    def __rtruediv__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.truediv,
            other,
            self.node,
        )

    def __floordiv__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.floordiv,
            self.node,
            other,
        )

    def __rfloordiv__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(
            operator.floordiv,
            other,
            self.node,
        )

    def __eq__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(operator.eq, self.node, other)

    def __ne__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(operator.ne, self.node, other)

    def __lt__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(operator.lt, self.node, other)

    def __gt__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(operator.gt, self.node, other)

    def __le__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(operator.le, self.node, other)

    def __ge__(self, other: Union[Self, Any]) -> Self:
        return self.node.create(operator.ge, self.node, other)

    def __index__(self) -> Self:
        return self.node.create(operator.index, self.node)

    def __len__(self) -> Self:
        return self.node.create(
            len,
            self.node,
        )

    ### Hacks ##############################

    def __iter__(self) -> Iterator[Self]:

        if not CONFIG.APP.CONTROL_FLOW_HANDLING:
            raise Exception(
                'Iteration control flow encountered but "CONFIG.APP.CONTROL_FLOW_HACKS" is set to False'
            )

        from ..hacks import iterator

        return iterator.handle_proxy(inspect.currentframe().f_back, self)

    def __bool__(self) -> Self:

        if not CONFIG.APP.CONTROL_FLOW_HANDLING:
            raise Exception(
                'Conditional control flow encountered but "CONFIG.APP.CONTROL_FLOW_HACKS" is set to False'
            )

        from ..hacks import conditional

        return conditional.handle_proxy(inspect.currentframe().f_back, self)

    def __instancecheck__(self, __instance: Any) -> bool:
        return self.node.fake_value.__instancecheck__(__instance)


ProxyType = TypeVar("ProxyType", bound=Proxy)


def proxy_patch(fn: Callable):

    @wraps(fn)
    def inner(*args, **kwargs):

        found: Proxy = None

        def find(proxy: Proxy):

            nonlocal found

            found = proxy

        util.apply((args, kwargs), find, Proxy)

        if found is not None:

            return found.node.graph.create(
                fn,
                *args,
                **kwargs,
            )

        return fn(*args, **kwargs)

    return inner

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Collection, TypeVar, Union

import torch
from typing_extensions import Self

from ... import util
from ...tracing.graph import Proxy
from .. import protocols

if TYPE_CHECKING:
    from . import InterventionNode


class InterventionProxy(Proxy):
        
    
    def __init__(self, node: "InterventionNode") -> None:
        super().__init__(node)

        self.__dict__["_grad"] = None

        self._grad: Self
        self.node: "InterventionNode"
    
    @property
    def grad(self) -> Self:
        """
        Calling denotes the user wishes to get the grad of proxy tensor and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: Grad proxy.
        """

        self.__dict__["_grad"] = protocols.GradProtocol.add(self.node.graph, self.node, fake_value=self.node.fake_value)

        return self._grad

    @grad.setter
    def grad(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the grad of this proxy tensor and therefore we create a Proxy of that request via a SwapProtocol.

        Args:
            value (Union[InterventionProxy, Any]): Value to set output to.
        """
        protocols.SwapProtocol.add(self.node.graph, self._grad, value)
        
    def __setattr__(
        self, key: Union[InterventionProxy, Any], value: Union[Self, Any]
    ) -> None:

        # We catch setting .grad as that is a special Protocol vs. setting attributes generally.
        if key == "grad":
            return getattr(self.__class__, key).fset(self, value)

        return super().__setattr__(key, value)

    @property
    def shape(self) -> Collection[torch.Size]:
        """Property to retrieve the shape of the traced proxy value or real value.

        Returns:
            Union[torch.Size,Collection[torch.Size]]: Proxy value shape or collection of shapes.
        """

        if not self.node.attached:

            return util.apply(self.value, lambda x: x.shape, torch.Tensor)

        # If we haven't scanned in a proxy_value, just return a proxy to get the attribute.
        if self.node.fake_value is inspect._empty:

            return super().__getattr__("shape")

        return util.apply(self.node.fake_value, lambda x: x.shape, torch.Tensor)

    @property
    def device(self) -> Collection[torch.device]:
        """Property to retrieve the device of the traced proxy value or real value.

        Returns:
            Union[torch.Size,Collection[torch.device]]: Proxy value device or collection of devices.
        """

        if not self.node.attached:

            return util.apply(self.value, lambda x: x.device, torch.Tensor)

        # If we haven't scanned in a proxy_value, just return a proxy to get the attribute.
        if self.node.fake_value is inspect._empty:

            return super().__getattr__("device")

        return util.apply(self.node.fake_value, lambda x: x.device, torch.Tensor)

    @property
    def dtype(self) -> Collection[torch.device]:
        """Property to retrieve the dtype of the traced proxy value or real value.

        Returns:
            Union[torch.Size,Collection[torch.dtype]]: Proxy value dtype or collection of dtypes.
        """

        if not self.node.attached:

            return util.apply(self.value, lambda x: x.dtype, torch.Tensor)

        # If we haven't scanned in a proxy_value, just return a proxy to get the attribute.
        if self.node.fake_value is inspect._empty:

            return super().__getattr__("dtype")

        return util.apply(self.node.fake_value, lambda x: x.dtype, torch.Tensor)
    
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

        util.apply((args, kwargs), get_proxy, Proxy)
    
        return proxy.node.create(
            orig_method,
            *args,
            **kwargs,
        )
    
    
InterventionProxyType = TypeVar("InterventionProxyType", bound=InterventionProxy)
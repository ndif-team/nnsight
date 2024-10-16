
import inspect
from collections import defaultdict
from contextlib import AbstractContextManager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from torch.utils.hooks import RemovableHandle
from typing_extensions import Self

from .. import util
from ..tracing import protocols
from ..tracing.graph import Graph, MultiGraph
from ..tracing.graph import Node
from .tracing.protocols import Protocol, GradProtocol
from ..tracing.graph import Proxy
from .tracing.util import backwards_check

class InterventionProxy(Proxy):
    """Sub-class for Proxy that adds additional user functionality to proxies.

    Examples:

        Saving a proxy so it is not deleted at the completion of it's listeners is enabled with ``.save()``:

        .. code-block:: python

            with model.trace('The Eiffel Tower is in the city of'):
                hidden_states = model.lm_head.input.save()
                logits = model.lm_head.output.save()

            print(hidden_states)
            print(logits)
    """

    def __init__(self, node: Node) -> None:
        super().__init__(node)

        self.__dict__["_grad"] = None

        self._grad: InterventionProxy

    def save(self) -> InterventionProxy:
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

    def local(self) -> InterventionProxy:
        """Streams value of this node locally when it becomes available remotely.
        This then kicks off execution of the local intervention graph up until it hits an upload Node created from `remote()`.

        Is a no-op when not executing remotely.

        Returns:
            InterventionProxy: Proxy.
        """

        return protocols.StreamingDownloadProtocol.add(self.node)

    def remote(self) -> InterventionProxy:
        """Streams value of this node remotely when it becomes available locally.
        The remote service will block until the local value is uploaded and received.

        Is a no-op when not executing remotely.

        Returns:
            InterventionProxy: Proxy.
        """

        return protocols.StreamingUploadProtocol.add(self.node.graph, self.node)

    def stop(self) -> None:
        """Method when called, indicates to the intervention graph to stop the execution of the model after this Proxy/Node is completed.."""

        protocols.EarlyStopProtocol.add(self.node.graph, self.node)

    def update(self, value: Union[Node, Any]) -> InterventionProxy:
        """Updates the value of the Proxy via the creation of the UpdateProtocol node.

        Args:
            - value (Union[Node, Any]): New proxy value.

        Returns:
            InterventionProxy: Proxy.

        .. codeb-block:: python
            with model.trace(input) as tracer:
                num = tracer.apply(int, 0)
                num.update(5)
        """

        return protocols.UpdateProtocol.add(self.node, value)

    @property
    def grad(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the grad of proxy tensor and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: Grad proxy.
        """

        self.__dict__["_grad"] = protocols.GradProtocol.add(self.node)

        return self._grad

    @grad.setter
    def grad(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the grad of this proxy tensor and therefore we create a Proxy of that request via a SwapProtocol.

        Args:
            value (Union[InterventionProxy, Any]): Value to set output to.
        """
        protocols.SwapProtocol.add(self.grad.node, value)

        self.__dict__["_grad"] = None

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

        if not self.node.attached():

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

        util.apply(args, get_proxy, Proxy)

        return proxy.node.create(
            orig_method,
            *args,
            **kwargs,
        )
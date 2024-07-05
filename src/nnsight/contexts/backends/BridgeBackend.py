from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from . import Backend

if TYPE_CHECKING:
    from ...tracing.Bridge import Bridge


class BridgeMixin:
    """To be inherited by objects that want to be able to be executed by the BridgeBackend."""

    def bridge_backend_handle(self, bridge: "Bridge") -> None:
        """Should add self to the current Bridge in some capacity.

        Args:
            bridge (Bridge): Current Bridge.
        """

        raise NotImplementedError()


class BridgeBackend(Backend):
    """Backend to accumulate multiple context object to be executed collectively.

    Context object must inherit from BridgeMixin and implement its methods.

    Attributes:

        bridge (Bridge): Current Bridge object.
    """

    def __init__(self, bridge: "Bridge") -> None:

        self.bridge = bridge

    def __call__(self, obj: BridgeMixin):

        obj.bridge_backend_handle(self.bridge)

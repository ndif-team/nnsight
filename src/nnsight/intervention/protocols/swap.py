from typing import TYPE_CHECKING, Any

import torch
from ...tracing.protocols import Protocol
from ... import util
if TYPE_CHECKING:
    from ..graph import InterventionNodeType, InterventionGraph, InterventionProxyType


class SwapProtocol(Protocol):
    

    @classmethod
    def execute(cls, node: "InterventionNodeType") -> None:
        
        intervention_node, value = node.args
        intervention_node: "InterventionNodeType"
        
        value = node.prepare_inputs(value)
        
        node.set_value(None)
        
        intervention_node.kwargs['swap'] = value

    @classmethod
    def get_swap(cls, graph: "InterventionGraph", value: Any) -> Any:
        """Checks if a swap exists on a Graph. If so get and return it, otherwise return the given value.

        Args:
            graph (Graph): Graph
            value (Any): Default value.

        Returns:
            Any: Default value or swap value.
        """

        # Tries to get the swap.
        swap: "InterventionNodeType" = graph.swap

        # If there was one:
        if swap is not None:

            device = None

            def _device(value: torch.Tensor):
                nonlocal device

                device = value.device

            # Get device of default value.
            util.apply(value, _device, torch.Tensor)

            # Get swap Node's value.
            value = util.apply(swap.args[1], lambda x: x.value, type(swap))

            if device is not None:

                def _to(value: torch.Tensor):
                    return value.to(device)

                # Move swap values to default value's device.
                value = util.apply(value, _to, torch.Tensor)

            # Set value of 'swap' node so it destroys itself and listeners.
            swap.set_value(True)

            # Un-set swap.
            graph.swap = None

        return value
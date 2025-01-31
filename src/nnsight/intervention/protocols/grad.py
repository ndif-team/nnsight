from typing import TYPE_CHECKING, Any, Dict

import torch

from ...tracing.protocols import Protocol

if TYPE_CHECKING:
    from ..graph import InterventionNode, InterventionNodeType
    
class GradProtocol(Protocol):
    """Protocol which adds a backwards hook via .register_hook() to a Tensor. The hook injects the gradients into the node's value on hook execution.
    Nodes created via this protocol are relative to the next time .backward() was called during tracing allowing separate .grads to reference separate backwards passes:

    .. code-block:: python
        with model.trace(...):

            grad1 = model.module.output.grad.save()

            model.output.sum().backward(retain_graph=True)

            grad2 = model.module.output.grad.save()

            model.output.sum().backward()

    Uses an attachment to store number of times .backward() has been called during tracing so a given .grad hook is only value injected at the appropriate backwards pass.
    """
        
    @classmethod
    def execute(cls, node: "InterventionNode") -> None:

        args, kwargs = node.prepare_inputs((node.args, node.kwargs))

        # First arg is the Tensor to add hook to.
        tensor: torch.Tensor = args[0]

        # Hook to remove when hook is executed at the appropriate backward pass.
        hook = None

        def grad(value):
               
            # Set the value of the Node.
            node.set_value(value)

            node.graph.execute(start=node.kwargs['start'], grad=True)
            
            # There may be a swap Protocol executed during the resolution of this part of the graph.
            # If so get it and replace value with it.
            if 'swap' in node.kwargs:
                value:InterventionNodeType = node.kwargs.pop('swap')
                
            # Remove hook (if this is not done memory issues occur)
            hook.remove()

            return value
            
        # Register hook.
        hook = tensor.register_hook(grad)

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        default_style = super().style()

        default_style["node"] = {"color": "green4", "shape": "box"}
    
        return default_style

        
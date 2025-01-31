from typing import TYPE_CHECKING, Any, Dict

import torch
from typing_extensions import Self

from ... import util
from ...tracing.protocols import Protocol

if TYPE_CHECKING:
    from ..graph import InterventionGraph, InterventionNode

class ApplyModuleProtocol(Protocol):
    """Protocol that references some root model, and calls its .forward() method given some input.
    Using .forward() vs .__call__() means it wont trigger hooks.
    Uses an attachment to the Graph to store the model.
    """


    @classmethod
    def add(
        cls, graph: "InterventionGraph", module_path: str, *args, hook=False, **kwargs
    ) -> Self:
        """Creates and adds an ApplyModuleProtocol to the Graph.
        Assumes the attachment has already been added via ApplyModuleProtocol.set_module().

        Args:
            graph (Graph): Graph to add the Protocol to.
            module_path (str): Module path (model.module1.module2 etc), of module to apply from the root module.

        Returns:
            InterventionProxy: ApplyModule Proxy.
        """
        
        from ..graph.node import ValidatingInterventionNode, validate

        # If the Graph is validating, we need to compute the proxy_value for this node.
        if graph.node_class is ValidatingInterventionNode:

            # If the module has parameters, get its device to move input tensors to.
            module: torch.nn.Module = util.fetch_attr(
                graph.model._model, module_path
            )

            try:
                device = next(module.parameters()).device
            except:
                device = None

            # Enter FakeMode for proxy_value computing.
            kwargs['fake_value'] = validate(module.forward, *args, **kwargs)

        kwargs["hook"] = hook

        # Create and attach Node.
        return graph.create(
            cls,
            module_path,
            *args,
            **kwargs,
        )

    @classmethod
    def execute(cls, node: "InterventionNode") -> None:
        """Executes the ApplyModuleProtocol on Node.

        Args:
            node (Node): ApplyModule Node.
        """
        
        graph: InterventionGraph = node.graph
    
        module: torch.nn.Module = util.fetch_attr(
            graph.model._model, node.args[0]
        )

        try:
            device = next(module.parameters()).device
        except:
            device = None

        args, kwargs = node.prepare_inputs((node.args, node.kwargs), device=device)

        module_path, *args = args

        hook = kwargs.pop("hook")

        if hook:
            output = module(*args, **kwargs)
        else:
            output = module.forward(*args, **kwargs)

        node.set_value(output)

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        default_style = super().style()

        default_style["node"] = {"color": "green4", "shape": "polygon", "sides": 6}

        return default_style

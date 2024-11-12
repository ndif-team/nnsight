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

   
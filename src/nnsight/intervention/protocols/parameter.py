from typing import Dict, Any

import torch

from ... import util
from ...tracing.protocols import Protocol

class ParameterProtocol(Protocol):
    """ Protocol designed for safe access of model tensor parameter attributes (e.g. weights and biases).

    When a model attribute that's a tensor is accessed, the attribute value is clone before being returned.
    """

    @classmethod
    def execute(cls, node):

        module: torch.nn.Module = util.fetch_attr(
            node.graph.model._model, node.args[0]
        )

        attr = getattr(module, node.args[1]).clone()

        node.set_value(attr)

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        default_style = super().style()

        default_style["node"] = {"color": "green4", "shape": "box"}

        default_style["arg_kname"][0] = "module_path"
        default_style["arg_kname"][1] = "key"

        return default_style
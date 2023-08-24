from __future__ import annotations

from typing import Any, Optional, Union

import torch.futures
import torch.fx


class TensorProxy(torch.fx.Proxy):
    @property
    def shape(self) -> torch.Size:
        """_summary_

        Returns:
            torch.Size: _description_
        """
        return self.tracer.node_name_to_value[self.node.name].shape

    @property
    def device(self):
        return self.tracer.node_name_to_value[self.node.name].device

    @property
    def dtype(self):
        return self.tracer.node_name_to_value[self.node.name].dtype

    def size(self, dim: int=None):
        if dim is None:
            return self.tracer.node_name_to_value[self.node.name].size()
        return self.tracer.node_name_to_value[self.node.name].size(dim)


class InterventionProxy(torch.futures.Future, TensorProxy):
    """
    We extend the Proxy class here for a couple reasons.
    We add the .save() method to denote were creating a save Node for a future Intervention and save a reference to it.
    We add torch.future.Future as a super class so we can set the value after Inference.
    """

    @staticmethod
    def proxy_set(
        activation_node: torch.fx.node.Node,
        value: Union[torch.fx.node.Node, Any],
    ) -> None:
        """
        Shell function to capture when were setting the value of a module output.
        activation_node is not used but it is an argument here so it is added to its dependencies
        (dont want to set the module output until weve actually arrived at the module).

        Args:
            activation_node (torch.fx.node.Node): _description_
            value (Union[torch.fx.node.Node, Any]): _description_
        """
        pass

    @staticmethod
    def proxy_save(node: torch.fx.node.Node) -> None:
        """Shell function to capture when were saving a value during Inference.

        Args:
            node (torch.fx.node.Node): _description_
        """
        pass

    def __init__(self, *args, **kwargs):
        torch.fx.Proxy.__init__(self, *args, **kwargs)
        torch.futures.Future.__init__(self)

    def set(self, value: Union[InterventionProxy, Any]):
        self.tracer.proxy(
            self.tracer.create_node(
                "call_function",
                InterventionProxy.proxy_set,
                (
                    self.node,
                    value.node if isinstance(value, InterventionProxy) else value,
                ),
                {},
            )
        )

    def save(self) -> InterventionProxy:
        """Creates a save proxy and adds it to Proxy.save_proxies."""
        proxy = self.tracer.proxy(
            self.tracer.create_node(
                "call_function", InterventionProxy.proxy_save, (self.node,), {}
            )
        )
        self.tracer.save_proxies[proxy.node.name] = proxy
        return proxy

    def token(self, idx: int) -> InterventionProxy:
        if idx >= 0:
            n_tokens = self.shape[1]
            idx = -(n_tokens - idx)

        return self[:, idx]

    def t(self, idx: int) -> InterventionProxy:
        return self.token(idx)


# When we use graph.eliminate_dead_code(), we want proxy_set and proxy_save and their dependencies to not be removed
torch.fx.node._side_effectful_functions.update(
    set([InterventionProxy.proxy_set, InterventionProxy.proxy_save])
)

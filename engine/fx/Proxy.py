from __future__ import annotations

from typing import Any, Union

import torch.futures
import torch.fx


class Proxy(torch.futures.Future, torch.fx.Proxy):
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

    def set(self, value: Union[Proxy, Any]):
        Proxy(
            self.tracer.create_node(
                "call_function",
                Proxy.proxy_set,
                (
                    self.node,
                    value.node if isinstance(value, Proxy) else value,
                ),
                {},
            ),
            self.tracer,
        )

    def save(self) -> Proxy:
        """Creates a save proxy and adds it to Proxy.save_proxies."""
        proxy = Proxy(
            self.tracer.create_node(
                "call_function", Proxy.proxy_save, (self.node,), {}
            ),
            self.tracer,
        )
        self.tracer.save_proxies[proxy.node.name] = proxy
        return proxy

    @property
    def shape(self) -> torch.Size:
        """_summary_

        Returns:
            torch.Size: _description_
        """
        return self.tracer.node_name_to_shape[self.node.name]

    def token(self, idx: int) -> Proxy:
        if idx >= 0:
            n_tokens = self.shape[1]
            idx = -(n_tokens - idx)

        return self[:, idx]

    def t(self, idx: int) -> Proxy:
        return self.token(idx)


# Set torch classes to our classes to account for other ways classes are created
torch.fx.proxy.Proxy = Proxy
# When we use graph.eliminate_dead_code(), we want proxy_set and proxy_save and their dependencies to not be removed
torch.fx.node._side_effectful_functions.update(set([Proxy.proxy_set, Proxy.proxy_save]))

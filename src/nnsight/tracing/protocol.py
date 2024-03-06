from typing import TYPE_CHECKING, Any, Dict, List, Type

import torch

from .Proxy import Proxy

if TYPE_CHECKING:
    from .Graph import Graph
    from .Node import Node


class Protocol:

    name: str

    @classmethod
    def add(cls, *args, **kwargs) -> Proxy:

        raise NotImplementedError()

    @classmethod
    def execute(cls, node: "Node"):

        raise NotImplementedError()


PROTOCOLS: Dict[str, Protocol] = dict()


def register_protocol(protocol: Type[Protocol]):

    PROTOCOLS[protocol.name] = protocol

    return protocol


@register_protocol
class ArgumentProtocol(Protocol):
    """Protocol never meant to be executed. Created node should be set by some outside force (very general).
    Graph has an `argument_node_names` attribute to find these by some name.
    """

    name = "arg"

    @classmethod
    def add(
        cls,
        graph: "Graph",
        name: str,
        value: Any,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
    ) -> Proxy:

        proxy = graph.add(value=value, target=cls.name, args=args, kwargs=kwargs)

        if name not in graph.argument_node_names:
            graph.argument_node_names[name] = []

        graph.argument_node_names[name].append(proxy.node.name)

        return proxy


@register_protocol
class LatchProtocol(Protocol):

    name = "latch"

    @classmethod
    def add(cls, node: "Node") -> Proxy:

        return node.add(
            value=None,
            target=cls.name,
            args=[node],
        )

    @classmethod
    def execute(cls, node: "Node") -> None:
        pass


@register_protocol
class GradProtocol(Protocol):

    name = "grad"

    @classmethod
    def add(cls, node: "Node") -> Proxy:

        # We track how many times backward is called via an attribute on the Graph
        if not hasattr(node.graph, "n_backward_calls"):

            setattr(node.graph, "n_backward_calls", 0)

        return node.add(
            value=node.proxy_value,
            target="grad",
            args=[node, node.graph.n_backward_calls],
        )

    @classmethod
    def execute(cls, node: "Node") -> None:

        args, kwargs = node.prepare_inputs()

        tensor: torch.Tensor = args[0]
        backward_idx: int = args[1]

        def grad(value):

            nonlocal backward_idx

            if backward_idx == 0:

                node.set_value(value)

                if node.is_tracing():

                    value = node.graph.get_swap(value)

                backward_idx = -1

                return value

            else:

                backward_idx -= 1

                return None

        tensor.register_hook(lambda value: grad(value))


@register_protocol
class SwapProtocol(Protocol):

    name = "swap"

    @classmethod
    def add(cls, node: "Node", value: Any) -> Proxy:

        return node.graph.add(target="swap", args=[node, value], value=True)

    @classmethod
    def execute(cls, node: "Node"):

        if node.graph.swap is not None:
            node.graph.swap.set_value(False)

        node.graph.swap = node

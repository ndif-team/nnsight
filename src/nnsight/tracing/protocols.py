import inspect
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .. import util
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


def register_protocol(protocol):

    PROTOCOLS[protocol.name] = protocol

    return protocol


@register_protocol
class ApplyModuleProtocol(Protocol):

    name = "module"
    attachment_name = "nnsight_root_module"

    @classmethod
    def add(cls, graph: "Graph", module_path: str, *args, **kwargs) -> Proxy:

        value = inspect._empty

        if graph.validate:

            try:

                module = cls.get_module(graph)
                device = next(module.parameters()).device

            except:

                device = None

            with FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(assume_static_by_default=True),
            ) as fake_mode:
                with FakeCopyMode(fake_mode):

                    value = cls.call_module(
                        graph,
                        module_path,
                        *Node.prepare_proxy_values(args, device=device),
                        **Node.prepare_proxy_values(kwargs, device=device),
                    )

        return graph.add(
            target=cls.name, value=value, args=[module_path] + list(args), kwargs=kwargs
        )

    @classmethod
    def execute(cls, node: "Node") -> None:

        args, kwargs = node.prepare_inputs()

        module_path, *args = args

        output = cls.call_module(node.graph, module_path, *args, **kwargs)

        node.set_value(output)

    @classmethod
    def call_module(cls, graph: "Graph", module_path: str, *args, **kwargs) -> Any:

        module = util.fetch_attr(cls.get_module(graph), module_path)

        return module.forward(*args, **kwargs)

    @classmethod
    def set_module(cls, graph: "Graph", module: torch.nn.Module) -> None:

        graph.attachments[cls.attachment_name] = module

    @classmethod
    def get_module(cls, graph: "Graph") -> torch.nn.Module:

        return graph.attachments[cls.attachment_name]


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
    attachment_name = "nnsight_backward_idx"

    @classmethod
    def add(cls, node: "Node") -> Proxy:

        backward_idx = node.graph.attachments.get(cls.attachment_name, 0)

        return node.add(
            value=node.proxy_value,
            target=cls.name,
            args=[node, backward_idx],
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

                    value = SwapProtocol.get_swap(node.graph, value)

                backward_idx = -1

                return value

            else:

                backward_idx -= 1

                return None

        tensor.register_hook(lambda value: grad(value))

    @classmethod
    def increment(cls, graph: "Graph"):

        backward_idx = graph.attachments.get(cls.attachment_name, 0)

        graph.attachments[cls.attachment_name] = backward_idx + 1


@register_protocol
class SwapProtocol(Protocol):

    name = "swap"
    attachment_name = "nnsight_swap"

    @classmethod
    def add(cls, node: "Node", value: Any) -> Proxy:

        return node.graph.add(target=cls.name, args=[node, value], value=True)

    @classmethod
    def execute(cls, node: "Node"):

        swap: "Node" = node.graph.attachments.get(cls.attachment_name, None)

        if swap is not None:
            swap.set_value(False)

        node.graph.attachments[cls.attachment_name] = node

    @classmethod
    def get_swap(cls, graph: "Graph", value: Any):

        swap: "Node" = graph.attachments.get(cls.attachment_name, None)

        if swap is not None:

            device = None

            def _device(value: torch.Tensor):
                nonlocal device

                device = value.device

            util.apply(value, _device, torch.Tensor)

            value = util.apply(swap.args[1], lambda x: x.value, type(swap))

            if device is not None:

                def _to(value: torch.Tensor):
                    return value.to(device)

                value = util.apply(value, _to, torch.Tensor)

            # Set value of 'swap' node so it destroys itself and listeners.
            swap.set_value(True)

            # Un-set swap.
            graph.attachments[cls.attachment_name] = None

        return value

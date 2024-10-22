from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Set, TypeVar, Union

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ... import util
from ...tracing.contexts import GlobalTracingContext
from ...tracing.graph import Node, Proxy
from ...tracing.protocols import Protocol
if TYPE_CHECKING:
    from . import InterventionGraph


class InterventionNode(Node):

    def __init__(
        self, *args, fake_value: Optional[Any] = inspect._empty, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.fake_value = fake_value

        self.graph: "InterventionGraph"

    @classmethod
    def prepare_inputs(
        cls, inputs: Any, device: Optional[torch.device] = None, fake: bool = False
    ) -> Any:

        inputs = util.apply(inputs, lambda x: x, inspect._empty)

        def inner(value: Union[InterventionNode, torch.Tensor]):

            nonlocal device

            if isinstance(value, InterventionNode):
                if fake:
                    value = value.fake_value
                else:
                    value = value.value

            if device is None and isinstance(value, torch.Tensor):
                device = value.device

            return value

        inputs = util.apply(
            inputs, inner, (InterventionNode, torch.Tensor), inplace=not fake
        )

        if device is not None:

            def _to(value: torch.Tensor):
                return value.to(device)

            inputs = util.apply(inputs, _to, torch.Tensor, inplace=not fake)

        return inputs


InterventionNodeType = TypeVar("InterventionNodeType", bound=InterventionNode)


class ValidatingInterventionNode(InterventionNode):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.fake_value is inspect._empty and (isinstance(self.target, type) and not issubclass(self.target, Protocol)):
            self.fake_value = validate(self.target, *self.args, **self.kwargs)


@staticmethod
def backwards_check(target: Callable, *args) -> bool:

    if target is Proxy.call:

        node: Node = args[0]

        if not isinstance(node, Node):
            return False

        if node.target is util.fetch_attr and node.args[1] == "backward":
            return True

    return False


@staticmethod
def validate(target: Callable, *args, **kwargs):

    # Enter FakeMode.
    with FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(assume_static_by_default=True),
    ) as fake_mode:
        with FakeCopyMode(fake_mode):

            with GlobalTracingContext.exit_global_tracing_context():

                if backwards_check(target, *args):
                    return None

                args, kwargs = InterventionNode.prepare_inputs(
                    (args, kwargs), fake=True
                )

                return target(
                    *args,
                    **kwargs,
                )

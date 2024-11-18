from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ... import util
from ...tracing.contexts import GlobalTracingContext
from ...tracing.graph import Node, Proxy
from ...tracing.protocols import Protocol
from ..protocols import EntryPoint

if TYPE_CHECKING:
    from . import InterventionGraph


class InterventionNode(Node):
    """This is the intervention extension of the base Node type.

    It has a fake_value to see information about this Node's future value before execution.
    It adds additional functionality to Node.prepare_inputs to handle Tensors.
    """

    def __init__(
        self, *args, fake_value: Optional[Any] = inspect._empty, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.fake_value = fake_value

    @classmethod
    def prepare_inputs(
        cls,
        inputs: Any,
        device: Optional[torch.device] = None,
        fake: bool = False,
    ) -> Any:
        """Override prepare_inputs to make sure

        Args:
            inputs (Any): _description_
            device (Optional[torch.device], optional): _description_. Defaults to None.
            fake (bool, optional): _description_. Defaults to False.

        Returns:
            Any: _description_
        """

        inputs = util.apply(inputs, lambda x: x, inspect._empty)

        def inner(value: Union[InterventionNode, torch.Tensor]):

            nonlocal device

            if isinstance(value, Proxy):
                value = value.node

            if isinstance(value, InterventionNode):
                if fake:
                    value = value.fake_value
                else:
                    value = value.value

            if device is None and isinstance(value, torch.Tensor):
                device = value.device

            return value

        inputs = util.apply(
            inputs, inner, (InterventionNode, Proxy, torch.Tensor), inplace=not fake
        )

        if device is not None:

            def _to(value: torch.Tensor):
                return value.to(device)

            inputs = util.apply(inputs, _to, torch.Tensor, inplace=not fake)

        return inputs

    def update_dependencies(self):
        for dependency in self.dependencies:
            if len(self.graph.defer_stack) > 0 and (
                dependency.index < self.graph.defer_stack[-1]
                or (
                    EntryPoint.is_entrypoint(dependency.target)
                    and dependency.graph is not self.graph
                )
            ):
                continue

            dependency.remaining_listeners -= 1

            if dependency.redundant:
                dependency.destroy()


InterventionNodeType = TypeVar("InterventionNodeType", bound=InterventionNode)


class ValidatingInterventionNode(InterventionNode):
    """The ValidatingInterventionNode executes its target using the fake_values of all of its dependencies to calculate a new fake_value for this node.
    Does not do this if the Node is detached from any graph, already has a fake_value (specified by whoever created the Node) or is a Protocol.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if (
            self.attached
            and self.fake_value is inspect._empty
            and not Protocol.is_protocol(self.target)
        ):
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

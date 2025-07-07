from typing import TYPE_CHECKING, Any, Callable

import torch

from ...util import Patch
from ..interleaver import Mediator
from .invoker import Invoker

if TYPE_CHECKING:
    from ..interleaver import Interleaver
else:
    Interleaver = Any


class BackwardsMediator(Mediator):

    def request(self, requester: Any):

        if not requester.endswith(".grad"):
            raise ValueError(
                f"Cannot request `{requester}` in a backwards tracer. You can only request `.grad`. Please define your Tensors before the Backwards Tracer and interact with their gradients within the Backwards Tracer."
            )

        return super().request(requester)


class BackwardsTracer(Invoker):

    def __init__(
        self,
        tensor: torch.Tensor,
        fn: Callable,
        interleaver: Interleaver,
        *args,
        **kwargs,
    ):

        super().__init__(None, *args, **kwargs)

        self.tensor = tensor
        self.fn = fn
        self.interleaver = interleaver

    def execute(self, fn: Callable):

        mediator = BackwardsMediator(fn, self.info)

        grad_patch = Patch(torch.Tensor, self.interleaver.wrap_grad(), "grad")

        self.interleaver.patcher.add(grad_patch)

        def inner():

            self.fn(self.tensor, *self.args, **self.kwargs)

            grad_patch.restore()

        self.interleaver.current.register(mediator, inner)

from typing import TYPE_CHECKING, Any, Callable

import torch

from ...util import Patch
from ..interleaver import Mediator, Interleaver
from .invoker import Invoker



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
        *args,
        **kwargs,
    ):

        super().__init__(None, *args, **kwargs)

        self.tensor = tensor
        self.fn = fn

    def execute(self, fn: Callable):

        mediator = BackwardsMediator(fn, self.info)

        
        interleaver = Interleaver([mediator], self)
        grad_patch = Patch(torch.Tensor, interleaver.wrap_grad(), "grad")
        
        try:

            with interleaver:
                interleaver.patcher.add(grad_patch)
                interleaver(self.fn, self.tensor, *self.args, **self.kwargs)
            self.push(interleaver.state)
            
        finally:
            interleaver.state.clear()

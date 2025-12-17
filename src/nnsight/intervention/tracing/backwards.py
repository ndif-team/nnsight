from typing import Any, Callable

import torch

from ...util import Patch
from ..interleaver import Interleaver, Mediator
from .invoker import Invoker


def wrap_grad(interleaver: Interleaver):
    """
    Create a hook for gradient intervention.

    Returns:
        A function that can be used to intercept gradients
    """

    def wrap(tensor: torch.Tensor):

        # Only wrap the tensor once
        if tensor._backward_hooks:
            return

        # We are providing the grad of the tensor
        provider = id(tensor)

        # Well need to remove the hook
        hook = None

        # On backwards for this tensor
        def inner(grad: torch.Tensor):

            # Inject the grad value
            # Possibly editing it in the process
            try:
                grad = interleaver.handle(f"{provider}.grad", grad)
            finally:
                hook.remove()

            return grad

        # Register the hook
        hook = tensor.register_hook(inner)

    def getter(tensor: torch.Tensor):

        wrap(tensor)

        requester = id(tensor)

        return interleaver.current.request(f"{requester}.grad")

    def setter(tensor: torch.Tensor, value: torch.Tensor):

        wrap(tensor)

        requester = id(tensor)

        return interleaver.current.swap(f"{requester}.grad", value)

    return property(getter, setter)


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

        grad_patch = Patch(torch.Tensor, wrap_grad(interleaver), "grad")

        try:
            grad_patch.patch()
            with interleaver:
                self.fn(self.tensor, *self.args, **self.kwargs)
            interleaver.check_dangling_mediators()

        finally:
            grad_patch.restore()
            interleaver.cancel()

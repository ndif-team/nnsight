from typing import Any, Callable, Optional, Tuple

import torch

from ...util import Patch
from ..interleaver import Interleaver, Mediator
from .invoker import Invoker


def _get_batch_narrow_info(
    tensor: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, int, int]]:
    """If *tensor* is a dim-0 narrow view of a base tensor, return
    ``(base, start, size)``.  Otherwise return ``None``.

    When multiple invokes are batched, ``Batcher._narrow`` slices the
    full-batch activation along dim 0 for each invoke.  The resulting
    view is **not** in the autograd forward path (the full tensor is
    returned to the next module), so ``register_hook`` on the view
    will never fire.  Detect such views so ``wrap_grad`` can register
    on the base tensor instead and narrow the gradient inside the hook.
    """
    base = tensor._base
    if base is None:
        return None
    if tensor.dim() == 0 or base.dim() == 0:
        return None
    if tensor.dim() != base.dim():
        return None
    if tensor.shape[1:] != base.shape[1:]:
        return None
    if tensor.stride() != base.stride():
        return None
    stride0 = base.stride()[0]
    if stride0 == 0:
        return None
    start = tensor.storage_offset() // stride0
    size = tensor.shape[0]
    if start + size > base.shape[0]:
        return None
    return (base, start, size)


def wrap_grad(interleaver: Interleaver):
    """
    Create a hook for gradient intervention.

    Returns:
        A function that can be used to intercept gradients
    """

    # Track which (hook_target_id, provider_id) pairs have been wrapped
    # so multiple narrow views of the same base each get their own hook.
    _wrapped: set = set()

    def wrap(tensor: torch.Tensor):

        # We are providing the grad of the tensor
        provider = id(tensor)

        # When the tensor is a batch-dim narrow view (created by
        # Batcher._narrow), the view is NOT in the autograd forward
        # path — the full-batch base tensor is what flows through the
        # model.  A hook on the view would never fire.  Register on the
        # base instead and narrow the gradient in the hook.
        narrow_info = _get_batch_narrow_info(tensor)

        if narrow_info is not None:
            hook_target, batch_start, batch_size = narrow_info
        else:
            hook_target = tensor
            batch_start = batch_size = None

        key = (id(hook_target), provider)
        if key in _wrapped:
            return
        _wrapped.add(key)

        # Well need to remove the hook
        hook = None

        # On backwards for this tensor
        def inner(grad: torch.Tensor):

            try:
                if batch_start is not None:
                    # Narrow the full-batch gradient to this invoke's
                    # slice, hand it to the mediator, and splice any
                    # user-supplied replacement back into the full grad.
                    narrow_grad = grad.narrow(0, batch_start, batch_size)
                    result = interleaver.handle(
                        f"{provider}.grad", narrow_grad
                    )
                    if result is not narrow_grad:
                        grad = grad.clone()
                        grad[batch_start : batch_start + batch_size] = result
                        return grad
                else:
                    grad = interleaver.handle(f"{provider}.grad", grad)
            finally:
                hook.remove()

            return grad

        # Register the hook and track it on the owning mediator so
        # Interleaver.cancel can clean it up if the worker thread dies
        # before the hook fires.
        hook = hook_target.register_hook(inner)
        mediator = interleaver.current
        if mediator is not None:
            mediator.hooks.append(hook)

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

from typing import Any, Callable, Optional, Tuple

import torch

from ...util import Patch
from ..interleaver import Interleaver, Mediator
from .invoker import Invoker

# Autograd raises a bare ``RuntimeError`` when a freed graph is walked again;
# there is no dedicated exception type or error code to match on.
_GRAPH_FREED_MARKER = "backward through the graph a second time"

_RETAIN_GRAPH_EXAMPLE = """    with model.trace() as tracer:
        with tracer.invoke(prompt_a):
            x = model.transformer.h[5].output[0]
            with model.lm_head.output.sum().backward(retain_graph=True):
                grad_a = x.grad.save()
        with tracer.invoke(prompt_b):
            y = model.transformer.h[5].output[0]
            with model.lm_head.output.sum().backward():
                grad_b = y.grad.save()"""


def _retain_graph_requested(args: Tuple[Any, ...], kwargs: dict) -> bool:
    """Whether this ``.backward()`` call asked for the graph to be retained.

    Mirrors ``Tensor.backward(gradient=None, retain_graph=None, ...)``, so
    ``retain_graph`` is the second positional parameter.
    """

    if "retain_graph" in kwargs:
        return bool(kwargs["retain_graph"])

    return len(args) > 1 and bool(args[1])


def _explain_freed_graph(
    error: RuntimeError, args: Tuple[Any, ...], kwargs: dict
) -> Optional[RuntimeError]:
    """Re-frame autograd's "graph freed" error in terms of nnsight's invokes.

    Every invoke in a trace contributes to *one* batched forward pass, so they
    also share one autograd graph. That makes a per-invoke ``.backward()`` look
    independent when it is not: the first one frees the graph and the next
    invoke fails inside ``torch.autograd``, with a message that never mentions
    invokes and so reads as unrelated to the code that triggered it.

    Returns ``None`` when the error is some other ``RuntimeError``, which the
    caller then re-raises untouched.
    """

    if _GRAPH_FREED_MARKER not in str(error):
        return None

    if _retain_graph_requested(args, kwargs):
        cause = (
            "This `.backward()` passed `retain_graph=True`, so an earlier "
            "`.backward()` in the same trace did not."
        )
    else:
        cause = (
            "An earlier `.backward()` in the same trace freed the graph that "
            "this one needs."
        )

    lines = [
        "The autograd graph for this trace has already been freed.",
        "",
        "Every invoke in a trace contributes to a single batched forward pass, "
        "so all invokes share one autograd graph. " + cause,
        "",
        "Pass `retain_graph=True` to every `.backward()` except the last one:",
        "",
        _RETAIN_GRAPH_EXAMPLE,
        "",
        "The same applies to two `.backward()` calls inside one invoke. If you "
        "do not need the gradients together, run each backward pass in its own "
        "trace instead.",
        "",
        f"Original error from torch.autograd: {error}",
    ]

    return RuntimeError("\n".join(lines))


def wrap_grad(interleaver: Interleaver):
    """
    Create a hook for gradient intervention.

    Returns:
        A function that can be used to intercept gradients
    """

    def wrap(tensor: torch.Tensor):

        # When two or more invokes share an input, the Batcher delivers each
        # invoke a storage-sharing view (`tensor.narrow(0, start, size)`) of the
        # full-batch activation. That view is never an input to any op producing
        # the loss, so a hook on it never fires. The Batcher tags such views with
        # their batch slice; redirect the hook to the full-batch parent and
        # narrow the gradient to this invoke's slice. The parent IS in the loss
        # graph, so its hook fires. (See Batcher._narrow in ../batching.py.)
        batch_group = getattr(tensor, "_nnsight_batch_group", None)
        redirect = batch_group is not None and tensor._base is not None

        # We are providing the grad of the tensor
        provider = id(tensor)

        # Well need to remove the hook
        hook = None

        if redirect:

            # Only wrap the view once.
            if getattr(tensor, "_nnsight_grad_wrapped", False):
                return

            target = tensor._base

            # The parent gradient has the same shape and (contiguous) layout as
            # the parent tensor, so this invoke's slice is recovered by re-applying
            # the view's own geometry. This is general: the parent may be the
            # flattened (batch*seq, hidden) activation, not (batch, seq, hidden).
            shape, stride, offset = tensor.shape, tensor.stride(), tensor.storage_offset()

            # On backwards for the parent tensor
            def inner(grad: torch.Tensor):

                # Slice out this invoke's gradient, let the user read/edit it,
                # and splice any edit back into the parent gradient.
                try:
                    sliced = grad.as_strided(shape, stride, offset)
                    new_sliced = interleaver.handle(f"{provider}.grad", sliced)
                    if new_sliced is not sliced:
                        grad = grad.clone()
                        grad.as_strided(shape, stride, offset).copy_(new_sliced)
                finally:
                    hook.remove()

                return grad

        else:

            # Only wrap the tensor once
            if tensor._backward_hooks:
                return

            target = tensor

            # On backwards for this tensor
            def inner(grad: torch.Tensor):

                # Inject the grad value
                # Possibly editing it in the process
                try:
                    grad = interleaver.handle(f"{provider}.grad", grad)
                finally:
                    hook.remove()

                return grad

        # Register the hook and track it on the owning mediator so
        # Interleaver.cancel can clean it up if the worker thread dies
        # before the hook fires.
        hook = target.register_hook(inner)
        if redirect:
            tensor._nnsight_grad_wrapped = True
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

        except RuntimeError as error:
            explained = _explain_freed_graph(error, self.args, self.kwargs)

            if explained is None:
                raise

            raise explained from error

        finally:
            grad_patch.restore()
            interleaver.cancel()

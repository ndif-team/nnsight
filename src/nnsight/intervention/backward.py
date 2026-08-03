"""Read and edit gradients during a backward pass.

Writing ``with tensor.backward():`` runs the real backward pass *interleaved*
with the body of the ``with`` block, so the block can read and replace the
``.grad`` of any tensor as the gradient reaches it.

A backward trace is almost always nested inside a forward trace, so the tensors
whose gradients you want are the real tensors produced during the run::

    with model.trace("The Eiffel Tower is in"):
        a1   = model.transformer.h[0].output
        loss = model.output.logits.sum()
        with loss.backward():
            g = a1.grad.save()          # capture the gradient flowing into a1
            a1.grad = a1.grad * 2       # ...and replace it
    print(g)

Gradients flow in the reverse of the forward pass, so ``.grad`` must be requested
in reverse-forward order — requesting an earlier-forward tensor's gradient before
a later one raises `OutOfOrderError`.
"""

from __future__ import annotations

import functools
from types import CodeType
from typing import Any, Callable

import torch

from ..tracing.tracer import Tracer, WithBlockNotFoundError, push_result
from ..tracing.util import shared_locals
from .interleaver import Interleaver, Mediator


def _grad_property(
    interleaver: Interleaver, seen: set[int], hooks: list
) -> property:
    """Build the ``Tensor.grad`` replacement property for one backward trace.

    The returned `property` is installed on ``torch.Tensor`` for the
    duration of a single backward run. Reading ``t.grad`` registers a
    self-removing autograd hook on ``t`` (at most once per tensor) and parks the
    block until autograd produces that gradient; assigning ``t.grad = v`` replaces
    the gradient that flows onward. Both operations are routed through the
    interleaver on the location ``f"{id(t)}.grad"``.

    Args:
        interleaver: The interleaver driving this backward trace; gradients are
            served to and swapped from the block through it.
        seen: Set of tensor ids already given a hook, used to avoid registering a
            second hook on the same tensor.
        hooks: List that collects the registered hook handles so
            [`BackwardTracer.execute`][nnsight.intervention.backward.BackwardTracer.execute] can remove them when the trace ends.

    Returns:
        property: A ``getter``/``setter`` property to bind to ``torch.Tensor.grad``.
    """

    def wrap(tensor: torch.Tensor) -> None:
        location = id(tensor)
        if location in seen:
            return
        seen.add(location)

        # A batched invoke reads a *view* of the full-batch activation (marked by
        # Batcher._narrow_tensor). That view is not in the loss graph — the model ran
        # on the full batch — so a hook on it never fires and the block would hang on
        # its gradient. Redirect the hook to the view's storage-owning base (which is
        # in the graph), recover exactly this view's elements from the base gradient
        # by its own strided geometry, and splice any edit back. Using the base (not a
        # stored parent tensor) keeps the marked view cheap to serialize.
        if getattr(tensor, "_nnsight_batch", False) and tensor._base is not None:
            base = tensor._base
            geometry = (tensor.shape, tensor.stride(), tensor.storage_offset())

            def hook(grad: torch.Tensor) -> torch.Tensor:
                try:
                    sliced = grad.as_strided(*geometry)
                    served = interleaver.handle(f"{location}.grad", sliced)
                    # A pure read returns the slice unchanged; an edit (swap) returns
                    # a new tensor to write back into the base gradient.
                    if served is sliced:
                        return grad
                    updated = grad.clone()
                    updated.as_strided(*geometry).copy_(served)
                    return updated
                finally:
                    handle.remove()

            handle = base.register_hook(hook)
            hooks.append(handle)
            return

        def hook(grad: torch.Tensor) -> torch.Tensor:
            # Serve (and maybe replace) the gradient, then stop intercepting this
            # tensor — one gradient flows per backward.
            try:
                return interleaver.handle(f"{location}.grad", grad)
            finally:
                handle.remove()

        handle = tensor.register_hook(hook)
        hooks.append(handle)

    def getter(tensor: torch.Tensor) -> Any:
        wrap(tensor)
        return Mediator.value(f"{id(tensor)}.grad")

    def setter(tensor: torch.Tensor, value: Any) -> None:
        wrap(tensor)
        Mediator.swap(f"{id(tensor)}.grad", value)

    return property(getter, setter)


class BackwardTracer(Tracer):
    """Read and edit gradients inside a ``with loss.backward():`` block.

    Opened by ``with loss.backward():`` (almost always nested inside a forward
    trace). Inside the block, read a tensor's incoming gradient with ``t.grad``,
    replace it with ``t.grad = ...``, and capture a value for use after the trace
    with ``.save()``. Gradients must be requested in reverse-forward order.

    Examples:
        >>> with model.trace("The Eiffel Tower is in"):
        ...     a1   = model.transformer.h[0].output
        ...     loss = model.output.logits.sum()
        ...     with loss.backward():
        ...         g = a1.grad.save()     # capture the gradient
        ...         a1.grad = a1.grad * 2  # ...and replace it
        >>> print(g)
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        fn: Callable,
        *args: Any,
        backend: Any = None,
        **kwargs: Any,
    ) -> None:
        """Store the ``backward`` call to replay under interleaving.

        Args:
            tensor: The tensor whose ``.backward(...)`` was invoked.
            fn: The real, unpatched ``Tensor.backward`` to call during execution.
            *args: Positional arguments forwarded to ``fn`` (e.g. ``gradient``).
            backend: Optional execution backend passed to the base
                `Tracer`.
            **kwargs: Keyword arguments forwarded to ``fn`` (e.g. ``retain_graph``).
        """
        super().__init__(backend=backend)
        self.tensor = tensor
        self.fn = fn  # the real Tensor.backward
        self.args = args
        self.kwargs = kwargs

    def execute(self, code: CodeType) -> None:
        """Run the real backward, serving the block's ``.grad`` reads and writes.

        Compiles the ``with`` block into an intervention mediator, installs the
        `_grad_property` on ``torch.Tensor``, and drives the real backward
        under an interleaver: as autograd produces each gradient, its hook hands
        the value to the block, which may read or replace it. Cleans up the patched
        property and every registered hook afterwards, then pushes the results back
        with save-gating (see [`Tracer.execute`][nnsight.tracing.tracer.Tracer.execute]).

        Args:
            code: Compiled code object for the body of the ``with`` block.
        """
        frame = self.info.frame

        # A tensor that doesn't require grad is outside autograd entirely — no
        # graph behind it, not itself an accumulation point — so no gradient
        # can flow from it to anything the block reads. (A requires-grad leaf
        # passes: grad_fn is None there too, but autograd still fires its
        # hook with the incoming gradient.)
        if not self.tensor.requires_grad:
            raise NotImplementedError(
                "This tensor does not require grad, so a backward session "
                "cannot produce gradients: nothing the block reads can ever "
                "receive one."
                + (
                    " This backend's forward runs under torch.inference_mode, "
                    "so gradients are unavailable here."
                    if torch.is_inference_mode_enabled()
                    else " The forward ran without gradient tracking (e.g. "
                    "under torch.no_grad()), or the tensor was created "
                    "without requires_grad=True."
                )
            )

        interleaver = Interleaver()
        mediator = Mediator(
            code, frame.f_globals, dict(frame.f_locals), shared=shared_locals(frame)
        )
        interleaver.mediators.append(mediator)

        seen: set[int] = set()
        hooks: list = []
        original = torch.Tensor.grad
        torch.Tensor.grad = _grad_property(interleaver, seen, hooks)
        try:
            # Real backward drives the run; the block's .grad reads/writes are
            # served by the autograd hooks as gradients flow. Autograd's engine
            # otherwise runs the graph on its own worker threads (e.g. on CUDA),
            # but the interventions run in a thread-bound greenlet — a hook firing
            # off-thread can't switch back into it. Force single-threaded backward
            # so every hook fires on this thread.
            with interleaver, torch.autograd.set_multithreading_enabled(False):
                self.fn(self.tensor, *self.args, **self.kwargs)
            interleaver.check_dangling_mediators()
            push_result(frame, mediator.lcls)
        finally:
            torch.Tensor.grad = original
            for handle in hooks:
                handle.remove()


_original_backward = torch.Tensor.backward


@functools.wraps(_original_backward)
def _backward(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    """Patched ``Tensor.backward`` that traces only when used as a context manager.

    ``with t.backward():`` returns a [`BackwardTracer`][nnsight.intervention.backward.BackwardTracer] whose ``__enter__``
    captures the following block. A bare ``t.backward()`` has no ``with`` block, so
    capture raises `WithBlockNotFoundError` and this falls
    through to the real backward — leaving ordinary usage unchanged.

    Args:
        tensor: The tensor bound as ``self`` for ``Tensor.backward``.
        *args: Positional arguments for ``backward`` (e.g. ``gradient``).
        **kwargs: Keyword arguments for ``backward`` (e.g. ``retain_graph``).

    Returns:
        A [`BackwardTracer`][nnsight.intervention.backward.BackwardTracer] when used as ``with t.backward():``, otherwise
        the return value of the real ``Tensor.backward`` (``None``).
    """
    tracer = BackwardTracer(tensor, _original_backward, *args, **kwargs)
    try:
        tracer.capture()
    except WithBlockNotFoundError:
        return _original_backward(tensor, *args, **kwargs)
    return tracer


def install() -> None:
    """Patch ``Tensor.backward`` so ``with t.backward():`` enters a BackwardTracer.

    Idempotent: replaces ``torch.Tensor.backward`` with `_backward` only if it
    isn't already installed. Called once at import time so the context-manager form
    is available everywhere, while plain ``t.backward()`` keeps working unchanged.
    """
    if torch.Tensor.backward is not _backward:
        torch.Tensor.backward = _backward


install()

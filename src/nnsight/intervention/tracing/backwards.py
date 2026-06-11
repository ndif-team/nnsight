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

        # Register the hook and track it on the owning mediator so
        # Interleaver.cancel can clean it up if the worker thread dies
        # before the hook fires.
        hook = tensor.register_hook(inner)
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

        from ..isolation import worker_backward_context

        ctx = worker_backward_context()
        if ctx is not None:
            return self._execute_isolated(fn, ctx)

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

    def _execute_isolated(self, fn: Callable, ctx: dict):
        """Run a backward block inside an isolated GPU worker.

        The autograd graph is split across the process boundary: the worker holds the
        recipe from each delivered activation up to ``self.tensor`` (the loss); the host
        holds the recipe from the model inputs up to those activations. We stitch the two
        halves at the named seam (the activation requester strings):

        1. The worker computes its half — ``dL/d(delivered activation)`` for each tagged
           leaf the loss depends on — as the seed.
        2. The host runs its half (``handle_backward_event``) seeded by those gradients
           and returns ``dL/d(activation)`` for every delivered activation, keyed by the
           same requester strings.
        3. The backward block body runs here in the worker; each ``.grad`` read is served
           from that returned dict by the tensor's provenance — no local ``register_hook``
           (the clones carry no host graph) and no local backward (the graph is host-side).
        """
        from ..interleaver import Events, Interleaver

        forward_mediator = ctx["mediator"]
        provenance = ctx["prov"]
        tagged = [t for t in ctx["tagged"] if t.requires_grad]

        # Worker half of the chain rule: seed = dL/d(delivered leaf) for leaves the loss
        # actually depends on (allow_unused drops the rest).
        seed: dict = {}
        if tagged:
            grads = torch.autograd.grad(
                self.tensor, tagged, allow_unused=True, retain_graph=True
            )
            for leaf, grad in zip(tagged, grads):
                if grad is not None:
                    seed[provenance[id(leaf)]] = grad

        # Host runs its half and returns dL/d(activation) keyed by requester string.
        worker_grads = forward_mediator.send(Events.BACKWARD, seed) or {}
        # The host signals "no graph at all" (forward ran without gradient tracking,
        # e.g. generate()) distinctly from "this particular read is off the path".
        no_graph = bool(worker_grads.pop("__nnsight_backward_no_graph__", False))

        mediator = BackwardsMediator(fn, self.info)
        interleaver = Interleaver([mediator], self)
        grad_patch = Patch(
            torch.Tensor,
            _isolated_grad_property(provenance, worker_grads, no_graph),
            "grad",
        )
        try:
            grad_patch.patch()
            # No local backward: the grads are already in worker_grads. The block body
            # just reads them (via the patched .grad) and .save()s; its saves push up
            # into the forward frame and ride the forward's END to the host.
            with interleaver:
                pass
            interleaver.check_dangling_mediators()
        finally:
            grad_patch.restore()
            interleaver.cancel()


def _isolated_grad_property(provenance: dict, worker_grads: dict, no_graph: bool = False):
    """A ``Tensor.grad`` property for the isolated backward block: read the gradient from
    the host-computed ``worker_grads`` by the tensor's delivery provenance (path), instead
    of registering a local autograd hook (the worker clone has no host graph)."""

    def getter(tensor: torch.Tensor):
        path = provenance.get(id(tensor))
        if path is None:
            raise RuntimeError(
                "gradient under isolation is only supported on an unmodified module-"
                "output tensor read during the trace (e.g. `model...ln_f.output`); "
                "this tensor was derived in user code and has no host-side graph."
            )
        if path not in worker_grads:
            if no_graph:
                raise RuntimeError(
                    f"no gradient available for `{path}` — the forward pass ran "
                    f"without gradient tracking, so there is no autograd graph "
                    f"(generate() runs grad-less; use model.trace() for gradients). "
                    f"This matches in-process behavior, where .grad raises here too."
                )
            raise RuntimeError(
                f"no gradient available for `{path}` — it is off the backward path "
                f"from the loss (its gradient never flowed during the backward pass)."
            )
        return worker_grads[path]

    def setter(tensor: torch.Tensor, value: Any):
        raise NotImplementedError(
            "editing `.grad` under isolation is not yet supported."
        )

    return property(getter, setter)

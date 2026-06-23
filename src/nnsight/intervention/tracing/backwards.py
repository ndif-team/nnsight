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

        worker_mediator = worker_backward_context()
        if worker_mediator is not None:
            return self._execute_isolated(fn, worker_mediator)

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

    def _execute_isolated(self, fn: Callable, worker_mediator):
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

        forward_mediator = worker_mediator
        provenance = worker_mediator._bwd_prov
        tagged = [t for t in worker_mediator._bwd_tagged if t.requires_grad]
        swaps = worker_mediator._bwd_swaps  # requester -> worker-tape swap value

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

        # Stitch the chain across the boundary, iterating over swap seams. Each round the
        # host returns dL/d(activation) for reads AND dL/d(swap leaf) (a swap installs a host
        # leaf, severing the host graph at the seam). The worker backprops each swap-leaf
        # grad through its swap tape to dL/d(delivered clone) and re-seeds the pre-swap graph
        # (another BACKWARD round). A read reached both directly and through a swap sums its
        # contributions across rounds. With no swaps this is the original single exchange.
        worker_grads: dict = {}
        no_graph = False
        while seed:
            resp = forward_mediator.send(Events.BACKWARD, seed) or {}
            no_graph = no_graph or bool(resp.pop("__nnsight_backward_no_graph__", False))
            swap_grads = resp.pop("__nnsight_swap_grads__", {})
            for path, g in resp.items():
                if torch.is_tensor(g):
                    worker_grads[path] = g if path not in worker_grads else worker_grads[path] + g
            # Next round's seed: backprop each returned swap-leaf grad through the worker's
            # swap tape to its delivered-clone leaves.
            seed = {}
            for swap_path, sg in swap_grads.items():
                swapped = swaps.get(swap_path)
                if not (tagged and torch.is_tensor(swapped) and torch.is_tensor(sg)):
                    continue
                grads = torch.autograd.grad(
                    swapped, tagged, grad_outputs=sg, allow_unused=True, retain_graph=True
                )
                for leaf, g in zip(tagged, grads):
                    if g is not None:
                        p = provenance[id(leaf)]
                        seed[p] = g if p not in seed else seed[p] + g

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

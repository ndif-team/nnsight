"""Per-(token, expert) rows of transformers' mixture-of-experts forwards.

A mixture-of-experts layer routes every token to ``top_k`` experts and sums their
weighted outputs. transformers' ``grouped_mm`` experts implementation runs all the
(token, expert) pairs of a forward as two grouped matrix multiplications over rows
sorted by expert: up projection, gate, down projection, routing weight, sum per
token. Its ``batched_mm`` implementation, which ``generate`` switches to for
decoding on an accelerator, does the same over the pairs in token order with one
weight matrix gathered per pair. The per-pair input, activation and down
projection are locals of those forwards; the module returns the sum.

Three sites on the experts module, each over the pairs in the token's own order,
in the dtype the forward computed them in:

* ``expert_input(tokens=, slot=)``: the row the pair's expert received.
* ``expert_activation(tokens=, slot=)``: the activation between the two
  projections (``act(gate) * up``, or ``act(up)`` for experts without a gate).
* ``expert_output(tokens=, slot=)``: the pair's down projection before its routing
  weight; the module's output is the sum over slots of these rows times the
  routing weights, cast to the input dtype.

Shapes are ``[batch, sequence, top_k, dim]``. ``expert=e`` (one int) in place of
``slot=`` gives the expert's rows per token, ``[batch, sequence, dim]``, ``0`` for a
token not routed to it. All three are exact exports: the forward's own rows, put
back in token order with its own inverse permutation where it sorted them. The
experts module computes over the flattened batch x sequence axis, as its
``.input`` and ``.output`` show; the reader folds that axis by the trace's batch
rows and hands each invoke its own, so ``tokens=`` indexes sequence positions
(a plain `Envoy` of the experts module counts its whole input as one row, so its
shapes are ``[1, tokens, ...]``; without any batcher the rows stay flat).

.. code-block:: python

    experts = model.model.layers[3].mlp.experts
    with model.trace(prompt):
        routed = experts.inputs[0][1].save()                                # top_k_index, [tokens, top_k]
        out = experts.kernel.expert_output(expert=7).save()                # [batch, seq, hidden]
        act = experts.kernel.expert_activation(tokens=-1).save()           # [batch, top_k, intermediate]
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch._subclasses.fake_tensor import FakeTensor

from . import kernel
from .kernel import Batch, Provenance, Sites, Unavailable
from .source import STATE, arguments, record

#: Per forward: the bindings holding the per-pair input, the activation (the gated
#: branch, or the plain one for experts without a gate) and the down projection.
#: `proj_out` is rebound at each stage, so the index is the stage's position in the
#: source; under expert parallelism the grouped forward also rebinds it masked, at
#: indices 1 and 5, which are not read.
_GROUPED = {"input": "selected_hidden_states_g", "gated": "proj_out_2", "ungated": "proj_out_3", "output": "proj_out_4"}
_BATCHED = {"input": "selected_hidden_states", "gated": "proj_out_1", "ungated": "proj_out_2", "output": "proj_out_3"}
#: Site name -> what its rows are, for provenance notes.
_ROWS = {"expert_input": "input rows", "expert_activation": "activation rows", "expert_output": "down-projection rows"}
#: Names transformers' tensor- and expert-parallel plans give the forward they install on an experts module.
_PARALLEL_FORWARDS = ("tp_forward", "ep_forward")


class Experts(Sites):
    """The per-pair rows of one experts forward, and the permutation back to token order when it sorted them."""

    names = tuple(_ROWS)

    def __init__(
        self,
        buffers: dict[str, torch.Tensor],
        inv_perm: torch.Tensor | None,
        top_k_index: torch.Tensor,
        num_experts: int,
        op_name: str,
    ) -> None:
        self.buffers = {name: value.detach() for name, value in buffers.items()}
        self.inv_perm = inv_perm
        self.top_k_index = top_k_index.detach()
        self.num_experts = num_experts
        self.op_name = op_name

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        if site not in self.names:
            raise ValueError(f"this kernel has no site {site!r}; available: {', '.join(self.names)}")
        return self._site(site, spec, batch)

    def _site(self, name: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        kernel.check_axes(spec, name, ("tokens", "slot", "expert"))
        if "slot" in spec and "expert" in spec:
            raise ValueError(f"kernel site {name!r} takes slot= or expert=, not both")
        tokens, top_k = self.top_k_index.shape
        # Pair (t, k) is row t * top_k + k of the forward's token-order rows; select
        # on these integer indices first, then gather only the rows asked for.
        pairs = torch.arange(tokens * top_k, device=self.top_k_index.device).view(tokens, top_k)
        index = self.top_k_index
        axis = 0
        if batch is not None:
            if tokens % batch.total:
                raise ValueError(
                    f"kernel site {name!r}: {tokens} token rows do not divide into the trace's {batch.total} batch rows, "
                    f"so they cannot be assigned to invokes"
                )
            length = tokens // batch.total
            pairs = pairs.view(batch.total, length, top_k)[batch.start : batch.start + batch.size]
            index = index.view(batch.total, length, top_k)[batch.start : batch.start + batch.size]
            axis = 1
        picked = kernel.indices(spec.get("tokens"), pairs.shape[axis], pairs.device)
        pairs, index = kernel.select(pairs, axis, picked), kernel.select(index, axis, picked)
        if "expert" in spec:
            expert = spec["expert"]
            if not isinstance(expert, int):
                raise TypeError(f"kernel site {name!r} takes one expert index; got {expert!r}")
            if not -self.num_experts <= expert < self.num_experts:
                raise IndexError(f"kernel site expert index {expert} is out of range for {self.num_experts} experts")
        else:
            slots = kernel.indices(spec.get("slot"), top_k, pairs.device)
            pairs, index = kernel.select(pairs, axis + 1, slots), kernel.select(index, axis + 1, slots)
        rows = self.buffers[name]
        kernel.warn_if_large((*pairs.shape, rows.shape[-1]), name, "tokens=, slot= or expert=", rows.dtype)
        value = rows[pairs if self.inv_perm is None else self.inv_perm[pairs]]
        if "expert" in spec:
            mask = (index == expert % self.num_experts).unsqueeze(-1)
            value = torch.where(mask, value, value.new_zeros(())).sum(axis + 1)
            value = kernel.squeeze(value, spec, {"tokens": axis})
            note = f"the expert's {_ROWS[name]} from the forward, 0 for a token not routed to it"
        else:
            value = kernel.squeeze(value, spec, {"slot": axis + 1, "tokens": axis})
            note = f"the forward's {_ROWS[name]} in token order; slot k is the token's k-th routed expert (top_k_index[..., k])"
        return value, Provenance("exact export", self.op_name, note)


def _parallel(module: torch.nn.Module) -> bool:
    """Whether transformers' tensor or expert parallelism installed its forward on ``module``.

    Under either, this rank's rows are its share (an intermediate shard and a
    partial down projection, or other ranks' pairs as sentinels). nnsight's
    controller sits in the instance ``forward`` and keeps the one it found there.
    """
    state = vars(module).get(STATE)
    original = state.original if state is not None else vars(module).get("forward")
    return module._is_expert_parallel or (original is not None and original.__name__ in _PARALLEL_FORWARDS)


def _serve(fn: Callable, args: tuple, kwargs: dict, layout: dict[str, str], sorted_rows: bool) -> tuple[Any, Sites]:
    output, bound = record(fn, args, kwargs, (*layout.values(), "inv_perm"))
    bound_arguments = arguments(fn, args, kwargs)
    module = bound_arguments["self"]
    if _parallel(module):
        return output, Unavailable(
            f"{fn.__name__} ran under tensor or expert parallelism; this rank's rows are its share of the experts, "
            f"so the per-pair sites are not served"
        )
    # Added to the forwards after transformers 5.19; the down projection is then not the last stage.
    if vars(module).get("has_post_expert_norm", False):
        return output, Unavailable(f"{fn.__name__} applies a post-expert norm; this reader covers the forward without one")
    index = bound_arguments["top_k_index"]
    pairs = index.numel()
    hidden = bound_arguments["hidden_states"].shape[-1]
    intermediate = module.down_proj.shape[-2 if module.is_transposed else -1]
    bindings = {
        "expert_input": layout["input"],
        "expert_activation": layout["gated"] if module.has_gate else layout["ungated"],
        "expert_output": layout["output"],
    }
    widths = {"expert_input": hidden, "expert_activation": intermediate, "expert_output": hidden}
    for site, binding in bindings.items():
        values = bound[binding]
        if len(values) != 1 or tuple(values[0].shape) != (pairs, widths[site]):
            return output, Unavailable(
                f"{fn.__name__} bound {binding} {len(values)} times with shapes {[tuple(v.shape) for v in values]}; "
                f"its forward no longer has the shape this reader expects"
            )
    inv_perm = None
    if sorted_rows:
        # `inv_perm` is bound empty and filled in place by a subscript assignment,
        # which the instrumentation records as a second binding of the assigned
        # value (the arange); the first binding is the filled permutation. Checked:
        # ordering the pairs by it must sort them by expert, as the forward's rows
        # are. Under `scan` the tensors are fake and carry no values to check.
        inv_perm = bound["inv_perm"][0] if bound["inv_perm"] else None
        if inv_perm is None or inv_perm.shape != (pairs,):
            return output, Unavailable(f"{fn.__name__} bound no inverse permutation over its {pairs} pairs")
        if not isinstance(inv_perm, FakeTensor) and not bool((index.reshape(-1)[inv_perm.argsort()].diff() >= 0).all()):
            return output, Unavailable(f"{fn.__name__} bound an inverse permutation that does not sort its {pairs} pairs by expert")
    buffers = {site: bound[binding][0] for site, binding in bindings.items()}
    return output, Experts(buffers, inv_perm, index, module.num_experts, fn.__name__)


def serve_grouped_mm(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """transformers' ``grouped_mm_experts_forward``: its expert-sorted per-pair rows and inverse permutation.

    The forward binds the sorted input as ``selected_hidden_states_g``, the
    activation and the down projection as rebindings of ``proj_out`` (see
    `_GROUPED`), and the inverse permutation as ``inv_perm``.
    """
    return _serve(fn, args, kwargs, _GROUPED, sorted_rows=True)


def serve_batched_mm(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """transformers' ``batched_mm_experts_forward``: its per-pair rows, already in token order.

    The forward binds the per-pair input as ``selected_hidden_states`` and the
    activation and down projection as rebindings of ``proj_out`` (see `_BATCHED`).
    """
    return _serve(fn, args, kwargs, _BATCHED, sorted_rows=False)

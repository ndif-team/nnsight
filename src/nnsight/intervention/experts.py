"""Per-(token, expert) rows from transformers' mixture-of-experts forwards.

A mixture-of-experts (MoE) layer routes each token to ``top_k`` experts and sums
their weighted outputs. In transformers, ``grouped_mm_experts_forward`` runs all
routed ``(token, expert)`` pairs across the forward pass using two grouped matrix
multiplications over rows sorted by expert ID (up/gate projection, activation,
down projection, routing-weight multiplication, and per-token sum). During
autoregressive decoding on an accelerator, ``generate`` switches to
``batched_mm_experts_forward``, which performs the same steps in token order by
gathering each pair's weight matrices. In both implementations, the per-pair
inputs, activations, and down-projected outputs exist only as local variables
inside the forward function, while the module returns only the weighted sum.

Each experts module exposes three sites over the routed pairs in token order,
returned in the dtype used by the forward pass:

* ``expert_input(tokens=, slot=)``: the input row received by each routed pair's
  expert.
* ``expert_activation(tokens=, slot=)``: the intermediate activation between the
  two projections (``act(gate) * up``, or ``act(up)`` for experts without a gate).
* ``expert_output(tokens=, slot=)``: the down-projected row before multiplying
  by the routing weight. Summing these rows multiplied by ``top_k_weights`` across
  the ``slot`` axis (and casting to the input dtype) reproduces the module's output.

Unsliced tensors have shape ``[batch, sequence, top_k, dim]``. Passing
``expert=e`` (a single integer) instead of ``slot=`` returns that expert's row
for each token of shape ``[batch, sequence, dim]``, filled with ``0`` at tokens
not routed to expert ``e``. All three sites are served as ``"exact export"``
tensors: when the forward sorted the pairs by expert ID, the site reorders them
back to token order using the forward's own inverse permutation ``inv_perm``.

Although the experts module itself operates on a flattened ``batch * sequence``
token axis (as seen on ``experts.input`` and ``experts.output``), the site reader
uses the trace's batch metadata to reshape the token axis into
``(batch, sequence)`` and slice out the requesting invoke's batch rows. As a
result, ``tokens=`` indexes sequence positions within each invoke. (When an
experts module is wrapped directly in a standalone `Envoy` without a batcher, or
when the module is the root of a single-row batch, shapes are
``[1, tokens, ...]`` or flat over tokens.)

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

#: Local variable binding names in `grouped_mm_experts_forward` and
#: `batched_mm_experts_forward` for the per-pair input, the gated or ungated
#: activation, and the down projection. Because `proj_out` is rebound at each
#: stage, the numeric suffix selects the stage's binding index in the AST.
#: Under expert parallelism, `grouped_mm_experts_forward` adds masked rebindings
#: at indices 1 and 5, which are unused because parallel experts return `Unavailable`.
_GROUPED = {"input": "selected_hidden_states_g", "gated": "proj_out_2", "ungated": "proj_out_3", "output": "proj_out_4"}
_BATCHED = {"input": "selected_hidden_states", "gated": "proj_out_1", "ungated": "proj_out_2", "output": "proj_out_3"}
#: Human-readable description of each site's rows for provenance notes.
_ROWS = {"expert_input": "input rows", "expert_activation": "activation rows", "expert_output": "down-projection rows"}
#: Function names installed on `experts.forward` by transformers' tensor-parallel and expert-parallel plans.
_PARALLEL_FORWARDS = ("tp_forward", "ep_forward")


class Experts(Sites):
    """Site reader holding the captured per-pair buffers of one experts forward pass."""

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
        # Pair (t, k) sits at linear index `t * top_k + k` in token order. Slice
        # this integer index table first so we only gather the requested rows from the buffer.
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
    """Return whether tensor parallelism or expert parallelism is active on ``module``.

    Under tensor or expert parallelism, each rank computes only a shard of the
    intermediate dimension and a partial down projection, or fills unowned pairs
    with sentinel indices. Because nnsight's controller wraps ``module.forward``,
    check the underlying ``state.original`` callable when present.
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
    # Guard against future transformers versions that add a post-expert norm after the down projection.
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
        # `inv_perm` is allocated empty and populated in place via subscript
        # assignment (`inv_perm[perm] = torch.arange(...)`). AST instrumentation
        # records the initial tensor at index 0 and the assigned `arange` at
        # index 1, so `bound["inv_perm"][0]` holds the completed inverse
        # permutation. Verify that sorting by its inverse orders pairs by expert ID
        # (skipped under `.scan()`, where FakeTensors have no concrete values).
        inv_perm = bound["inv_perm"][0] if bound["inv_perm"] else None
        if inv_perm is None or inv_perm.shape != (pairs,):
            return output, Unavailable(f"{fn.__name__} bound no inverse permutation over its {pairs} pairs")
        if not isinstance(inv_perm, FakeTensor) and not bool((index.reshape(-1)[inv_perm.argsort()].diff() >= 0).all()):
            return output, Unavailable(f"{fn.__name__} bound an inverse permutation that does not sort its {pairs} pairs by expert")
    buffers = {site: bound[binding][0] for site, binding in bindings.items()}
    return output, Experts(buffers, inv_perm, index, module.num_experts, fn.__name__)


def serve_grouped_mm(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Serve ``grouped_mm_experts_forward`` by capturing its expert-sorted pair buffers and ``inv_perm`` (see `_GROUPED`)."""
    return _serve(fn, args, kwargs, _GROUPED, sorted_rows=True)


def serve_batched_mm(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Serve ``batched_mm_experts_forward`` by capturing its token-ordered pair buffers (see `_BATCHED`)."""
    return _serve(fn, args, kwargs, _BATCHED, sorted_rows=False)

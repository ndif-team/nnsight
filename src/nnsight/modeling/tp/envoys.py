"""Envoys for modules transformers split across ranks.

The ad-hoc-call half of what
[`TPFragments`][nnsight.modeling.tp.fragments.TPFragments] does for activations.
Interleaving already shows a worker whole activations — gathered on the way in,
re-split on the way out, once per visit — but a trace that calls a module
*directly*, away from its place in the forward pass, is outside that bracket. A
logit lens running ``lm_head`` on an intermediate hidden state is holding, and
wants back, whole tensors, while the sharded module deals in slices.

So the bracket is the same one vLLM's envoy uses
([`nnsight.modeling.vllm.envoys`][nnsight.modeling.vllm.envoys]): re-split the
caller's whole input, run the module, reassemble the output, all off
`TPFragments`' own rules.

What is *not* needed here is any handling of the collectives themselves.
[`Envoy.__call__`][nnsight.intervention.envoy.Envoy.__call__] runs the module the
ordinary way — standing the interleaver down rather than dodging ``__call__`` —
so the style's own transforms fire around the call, which is what makes a
row-parallel layer's all-reduce and ``colwise_gather_output``'s all-gather happen
without this module restating them.

It does mean the style's *input* transform runs too, which is why re-splitting
the input is conditional: see
[`SPLITS_ITS_OWN_INPUT`][nnsight.modeling.tp.envoys.SPLITS_ITS_OWN_INPUT].

Parameters are left alone. ``layer.weight`` is the ``DTensor`` transformers made
of it — this rank holds a slice, while ``.shape`` reports the whole — as it is
anywhere else under transformers tensor parallelism.
"""

from __future__ import annotations

from typing import Any

import torch

from ...intervention.envoy import Envoy
from .fragments import _gather, _placement

#: Styles whose own input transform splits the input for them.
#:
#: These expect a *whole* tensor at the module boundary, so a caller handing one
#: in has already given the transform what it wants and nnsight must not cut it
#: down first — the transform would then cut the slice again. Every other
#: sharded-input style receives an input the column-parallel layer upstream had
#: already split, passes it straight through, and so needs the caller's whole
#: tensor split here.
#:
#: Read off ``RowwiseParallel``, which redistributes when the input layout it
#: declares is not the one its sharded weight needs — true of exactly those
#: registered with ``input_layouts=Replicate()``. Plain ``rowwise`` declares
#: ``Shard(-1)`` and passes through.
SPLITS_ITS_OWN_INPUT = ("rowwise_split_input", "rowwise_rep")

#: Styles whose output is still this rank's slice *after* the module's own output
#: transform has run. An ad-hoc call runs that transform, so what it returns is
#: the finished value: a row-parallel output has been all-reduced and a
#: ``colwise_gather_output`` head gathered (both whole, nothing to do), while a
#: plain column-parallel output is never gathered by transformers and a
#: ``sequence_parallel`` one has just been reduce-scattered — both come back
#: sharded on the last dim. This is the one place that finished view is needed;
#: [`SIDES`][nnsight.modeling.tp.fragments.SIDES] describes the value at the
#: handoff, *before* the transform, and must not be consulted for the output here.
SHARDED_AFTER_CALL = ("colwise", "packed_colwise", "sequence_parallel")


class TPEnvoy(Envoy):
    """An envoy over a module transformers may have sharded.

    Inert on an unsharded model and on a one-rank mesh — `TPFragments` records
    a rule only for a module actually split — so this is a safe envoy class for
    every ``Linear`` and ``Embedding`` in a tree that was loaded across ranks.
    """

    def __call__(self, *args: Any, hook: bool = False, **kwargs: Any) -> Any:
        """Run this module ad hoc, on whole tensors either side.

        Every rank runs the block, so every rank reaches the same collectives in
        the same order — as long as the call is not under rank-dependent control
        flow, the condition every collective in a block carries.
        """
        fragments = self.interleaver.fragments
        if hook or fragments is None or not fragments.enabled:
            return super().__call__(*args, hook=hook, **kwargs)

        # Asked of the fragments rather than read off the module: up to 5.15
        # transformers stamped the style on each module it sharded, and 5.16
        # keeps the plan on the model and stamps nothing, so the tree's own
        # record is the only spelling that works on both.
        style, mesh = fragments.style_at(self.path)
        into = f"{self.path}.input"
        if fragments.fragmented(into) and style not in SPLITS_ITS_OWN_INPUT:
            # `split`, not the way back from a gather: the caller's tensor is
            # whole because they are holding it, not because anything assembled
            # it, and this call may be nested inside that location's own open
            # handoff.
            args, kwargs = fragments.split(into, (args, kwargs))

        # The module's own transforms run inside this — the interleaver stands
        # itself down rather than bypassing them — so the style's collectives
        # happen here without being restated.
        result = super().__call__(*args, hook=False, **kwargs)

        if style in SHARDED_AFTER_CALL:
            result = _gather(result, mesh, _placement("shard"))
        return result


def tp_envoys() -> dict:
    """The ``envoys`` map pairing shardable module types with `TPEnvoy`.

    Keyed by type rather than by style because a style belongs to the *instance*
    — the same ``nn.Linear`` class is ``colwise`` in one place and ``rowwise`` in
    another — while the map is consulted per module as the tree is built.
    `TPEnvoy` asks `TPFragments` which style its own path got.
    """
    return {
        torch.nn.Linear: TPEnvoy,
        torch.nn.Embedding: TPEnvoy,
    }


def wants_tensor_parallel(target: Any, load_kwargs: dict) -> bool:
    """Whether this construction is going to produce a sharded model.

    Asked from a constructor, before the model exists — so for a repo id it reads
    the *request* (``distributed_config``) rather than the result. A ready module
    is already there and can be asked directly.

    Erring towards True is the safe direction: `TPEnvoy` is inert on a module
    with no style stamped on it, so a false positive costs a different envoy
    class and nothing else, while a false negative silently leaves ad-hoc calls
    handing back slices. Hence ``tp_plan`` counts even with no ``tp_size``.
    """
    if isinstance(target, torch.nn.Module):
        # transformers keeps both on the model. `_tp_plan` alone proves nothing —
        # every model declares one whether or not it was loaded across ranks — so
        # the mesh is the test.
        mesh = getattr(target, "_device_mesh", None)
        return bool(
            getattr(target, "_tp_plan", None)
            and mesh is not None
            and getattr(mesh, "size", lambda: 1)() > 1
        )

    config = load_kwargs.get("distributed_config")
    if config is None:
        return False
    if getattr(config, "tp_plan", None):
        return True
    return (getattr(config, "tp_size", None) or 1) > 1

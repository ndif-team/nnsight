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
transformers keeps them in forward hooks rather than inside ``forward``
(``distribute_module`` registers the style's ``_prepare_input_fn`` as a pre-hook
and ``_prepare_output_fn`` as a post-hook), and
[`Envoy.__call__`][nnsight.intervention.envoy.Envoy.__call__] runs the module the
ordinary way — standing the interleaver down rather than dodging ``__call__`` —
so those hooks fire on their own. That is what makes a row-parallel layer's
all-reduce and ``colwise_gather_output``'s all-gather happen without this module
restating them.

It does mean the style's *pre*-hook fires too, which is why re-splitting the
input is conditional: see
[`SPLITS_ITS_OWN_INPUT`][nnsight.modeling.tp.envoys.SPLITS_ITS_OWN_INPUT].

Parameters are left alone. ``layer.weight`` is this rank's real slice, as it is
anywhere else under transformers tensor parallelism.
"""

from __future__ import annotations

from typing import Any

import torch

from ...intervention.envoy import Envoy
from .fragments import _gather

#: Styles whose own pre-hook splits the input for them.
#:
#: Both of these take a *whole* tensor at the module boundary, so a caller
#: handing one in has already given the hook what it wants and nnsight must not
#: cut it down first — the hook would then cut the slice again. Every other
#: sharded-input style (plain ``rowwise``, ``packed_rowwise``) receives an input
#: that was already split by the column-parallel layer upstream, so its pre-hook
#: passes the value straight through and the caller's whole tensor *is* what
#: needs splitting.
#:
#: Read off ``RowwiseParallel._prepare_input_fn``, which splits only when its
#: ``split_input`` is set — true for ``rowwise_split_input`` alone among the
#: styles transformers registers.
SPLITS_ITS_OWN_INPUT = ("rowwise_split_input",)

#: Styles whose output is still this rank's slice *after* the module's own
#: post-hook has run. An ad-hoc call runs those hooks, so the value it returns
#: is the post-hook one: a row-parallel output has been all-reduced and a
#: ``colwise_gather_output`` head gathered (both whole, nothing to do), while a
#: plain column-parallel output is never gathered by transformers and a
#: ``sequence_parallel`` one has just been reduce-scattered — both come back
#: sharded on the last dim. This is the one place the post-hook view is needed;
#: [`SIDES`][nnsight.modeling.tp.fragments.SIDES] describes the handoff's
#: pre-hook view and must not be consulted for the output here.
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

        module = self._module
        style = getattr(module, "_hf_tp_plan", None)
        into = f"{self.path}.input"
        if fragments.fragmented(into) and style not in SPLITS_ITS_OWN_INPUT:
            args, kwargs = fragments.fragment(into, (args, kwargs))

        # The module's own hooks run inside this — the interleaver stands itself
        # down rather than bypassing them — so the style's collectives happen
        # here without being restated.
        result = super().__call__(*args, hook=False, **kwargs)

        if style in SHARDED_AFTER_CALL:
            result = _gather(result, module._hf_device_mesh)
        return result


def tp_envoys() -> dict:
    """The ``envoys`` map pairing shardable module types with `TPEnvoy`.

    Keyed by type rather than by style because a style is stamped on the
    *instance* at load — the same ``nn.Linear`` class is ``colwise`` in one place
    and ``rowwise`` in another — while the map is consulted per module as the
    tree is built. `TPEnvoy` reads the stamp itself.
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
        return any(
            getattr(m, "_hf_tp_plan", None) is not None
            and getattr(getattr(m, "_hf_device_mesh", None), "size", lambda: 1)() > 1
            for m in target.modules()
        )

    config = load_kwargs.get("distributed_config")
    if config is None:
        return False
    if getattr(config, "tp_plan", None):
        return True
    return (getattr(config, "tp_size", None) or 1) > 1

"""When a value at a location is a piece of a larger one.

On a model split across devices, what a module hands back can be a fragment: one
rank's columns of a linear's output, one rank's partial sum, one rank's share of
an expert combine. A user asked for the layer, not a quarter of it, so a runtime
that shards has to make the value whole before intervention code sees it and put
it back before the model's own forward carries on.

Which values are fragments, and what "whole" means for each, is the *runtime's*
knowledge — transformers labels its shards one way, vLLM another, and the
collectives differ. Everything else about the bracket is the same regardless, so
that part lives here and in
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle],
which is the one place that sees a location's value exactly once per visit.

## Why once per visit matters

It is the whole reason this hangs off the interleaver rather than the batcher.

[`Batcher.narrow`][nnsight.intervention.batching.Batcher.narrow] is called **once
per mediator** — once for each worker parked on a location. A batcher that
gathered would therefore fire one collective per *reader*, and since which
workers are parked is a property of the user's block rather than of the model,
ranks would run different numbers of collectives and deadlock. Doing it in
``handle`` sidesteps that: one visit, one collective, however many workers read
it, with no memoization to keep in step.

## The rules a subclass must follow

* **Never branch on rank.** Every rank runs the same intervention block and must
  reach the same collectives in the same order. `fragmented` in particular is
  asked on every rank and must answer identically on all of them.
* **`whole`'s ``undo`` must reverse it.** What comes back goes into the model's
  own forward. A value passed through untouched must come back untouched — a
  tensor that was never a fragment must not be split.
* **`split` is not that operation.** It cuts down a value that was never
  gathered, so the location's rule is all it has to work from. Serving both from
  one method let the assembling path and the ad-hoc path be mistaken for each
  other, and a nested ad-hoc call consumed the record its caller was still using.
"""

from __future__ import annotations

from typing import Any


class Fragments:
    """A runtime's answer to "is this value a piece of a larger one".

    The base is the identity: nothing is a fragment, nothing is gathered, and an
    [`Interleaver`][nnsight.intervention.interleaver.Interleaver] carrying one of
    these behaves exactly as one carrying none. A distributed runtime subclasses
    it; everything else is unaffected.
    """

    #: Whether this tree has any fragments at all. False is the common case even
    #: for a runtime that *can* shard — a HuggingFace model is built with a
    #: `TPFragments` whether or not it was loaded across ranks — so it is checked
    #: before anything else and costs one attribute lookup per handled location.
    #: A subclass flips it in `instrument` on finding something actually split.
    enabled: bool = False

    def instrument(self, envoy: Any) -> None:
        """Learn what this envoy's values are, as the tree is built.

        Called for every envoy from
        [`Interleaver.instrument`][nnsight.intervention.interleaver.Interleaver.instrument],
        which is the one moment both the module — carrying whatever its runtime
        stamped on it — and its path are in hand. The handoff runs inside the
        module's forward, after its runtime's pre-hooks and before its post-hooks,
        so what is recorded here is what the value is at *that* point. Default:
        nothing to learn.
        """

    def fragmented(self, location: str) -> bool:
        """Whether the value at ``location`` is one piece of a larger tensor.

        Answered from the location alone, so it costs a dict lookup on the hot
        path and needs nothing from the value. Default: never.
        """
        return False

    def whole(self, location: str, value: Any) -> "tuple[Any, Any]":
        """The real value at ``location``, and how to put back what that took.

        Returns ``(whole, undo)``. ``undo`` is called with whatever intervention
        code left behind — so an edit to the assembled tensor is carried back into
        the model rather than discarded — or is ``None`` when there was nothing to
        assemble.

        Handed back rather than looked up again later because only this call knows
        what it did: where a value can describe its own layout, the way back
        depends on the value that arrived and not on the location alone. Holding
        that on the caller's stack is also what keeps a nested handoff — an ad-hoc
        call made while this visit is still open — from consuming it.

        Only called when `fragmented` said so *and* something is actually waiting
        to read it. Default: unreachable, since nothing is ever fragmented.
        """
        return value, None

    def split(self, location: str, whole: Any) -> Any:
        """This device's piece of an already-whole value, from the rule alone.

        For values that never came out of the model and so were never assembled: a
        `.skip` replacement, and the argument of an ad-hoc call on a sharded
        module. Both are the real tensor already, and both have to be cut down
        before the model's own forward sees them.
        """
        return whole

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
* **`fragment` must undo `whole`.** What comes back goes into the model's own
  forward. A value `whole` passed through untouched must come back untouched — a
  tensor that was never a fragment must not be split.
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
        stamped on it — and its path are in hand. Default: nothing to learn.
        """

    def begin(self) -> None:
        """Called as a run starts, for whatever is per-run rather than per-model.

        The rules themselves are a property of the loaded model and outlive a
        run; anything a *reader* should be told about is not. Default: nothing to
        reset.
        """

    def read(self, location: str) -> None:
        """Called when something is about to read ``location``.

        Only for locations something is actually waiting on, and regardless of
        whether they are fragments — which is what makes this the place to say
        something about a value the runtime *cannot* make whole. Default: nothing
        to say.
        """

    def fragmented(self, location: str) -> bool:
        """Whether the value at ``location`` is one piece of a larger tensor.

        Answered from the location alone, so it costs a dict lookup on the hot
        path and needs nothing from the value. Default: never.
        """
        return False

    def whole(self, location: str, value: Any) -> Any:
        """The real value at ``location``, assembled from every device's piece.

        Only called when `fragmented` said so *and* something is actually waiting
        to read it. Default: unreachable, since nothing is ever fragmented.
        """
        return value

    def fragment(self, location: str, whole: Any) -> Any:
        """This device's piece of ``whole``, as the model's own forward expects.

        The inverse of `whole`, called with whatever intervention code left
        behind — so an edit made to the assembled tensor is carried back into the
        model rather than discarded.
        """
        return whole

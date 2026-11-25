"""Run a module once for every worker that calls it at the same moment.

A block can call a module of the model on a value it holds, away from that
module's place in the forward pass: a logit lens runs the final norm and the
unembedding on an intermediate hidden state. When several workers share one
forward (a serving engine batching many requests), each of them makes that call
on its own rows, so the module runs once per worker on a small tensor.

With a [`Coalescer`][nnsight.intervention.coalesce.Coalescer] on the interleaver,
such a call parks the worker instead ([`Event.CALL`][nnsight.intervention.interleaver.Event]).
Once every worker being served has parked or moved on,
[`Interleaver.settle`][nnsight.intervention.interleaver.Interleaver.settle] hands
the parked calls here. Calls of one module with compatible arguments are run as
one call on the callers' rows concatenated, and each caller is resumed with its
own rows of the result.

That is only correct for a call that is token-wise: it computes each output row
from its own input row and leaves its arguments as they were. A norm and a
linear map are; anything that mixes rows (a mean over tokens, attention) is not,
and concatenating two requests' rows into it would let one request's values
reach another's. The coalescer does not find this out by running anything. The
runtime states which calls are token-wise when it builds the coalescer (the
``tokenwise`` predicate), and a module can be added by name with
[`register`][nnsight.intervention.coalesce.Coalescer.register]; any other call
runs on its caller's arguments alone.

Workers take turns at a module call: one parked on a call lets the next worker
run before it gets its result. A runtime whose workers have separate scopes and
separate rows (requests of a serving engine) is unaffected by that order. Invokes
of one local trace share a scope and can depend on running in order, so a local
model is given no coalescer.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, NamedTuple, Optional

import torch

from .envoy import Envoy
from .interleaver import Call

# Argument values compared by value when deciding whether two calls match.
PLAIN = (type(None), bool, int, float, str)


class Outcome(NamedTuple):
    """What a worker is resumed with: the call's result, or the error it raised."""

    value: Any = None
    error: Optional[BaseException] = None


class Unsplittable(Exception):
    """A result holds something that is not one row per input row."""


def split(result: Any, sizes: list[int]) -> list:
    """``result`` cut into one piece per caller, ``sizes[i]`` rows each.

    A tensor is cut along its first dimension, into views. A list, tuple or dict
    is cut leaf by leaf and rebuilt per caller. ``None`` is every caller's
    ``None``: a module can return nothing, and one that gathers its output onto
    a single rank of a distributed engine returns nothing on the others.

    Raises:
        Unsplittable: a leaf is neither, or a tensor does not have one row per
            input row.
    """
    if result is None:
        return [None] * len(sizes)
    if isinstance(result, torch.Tensor):
        if result.ndim == 0 or result.shape[0] != sum(sizes):
            raise Unsplittable()
        pieces, start = [], 0
        for size in sizes:
            pieces.append(result.narrow(0, start, size))
            start += size
        return pieces
    if isinstance(result, (list, tuple)):
        columns = [split(item, sizes) for item in result]
        rebuild = (
            (lambda items: type(result)(*items))
            if isinstance(result, tuple) and hasattr(result, "_fields")
            else type(result)
        )
        return [rebuild([column[i] for column in columns]) for i in range(len(sizes))]
    if isinstance(result, dict):
        columns = {key: split(item, sizes) for key, item in result.items()}
        pieces = []
        for i in range(len(sizes)):
            piece = result.copy()
            for key, column in columns.items():
                piece[key] = column[i]
            pieces.append(piece)
        return pieces
    raise Unsplittable()


class Group:
    """Calls of one module that can run as one call.

    ``rows`` names the positional and keyword arguments that are concatenated
    over the callers: tensors with one row per row of the caller's input, a
    different tensor for each caller. Every other argument is passed once, as
    the first caller gave it. ``sizes`` is each caller's row count.
    """

    def __init__(self, members: list, rows: list, sizes: list[int]) -> None:
        self.members = members
        self.rows = rows
        self.sizes = sizes

    @property
    def call(self) -> Call:
        return self.members[0][1]

    def arguments(self) -> tuple[list, dict]:
        """The one call's ``(args, kwargs)``."""
        calls = [call for _, call in self.members]

        def build(slot: Any) -> Any:
            if slot in self.rows:
                return torch.cat([argument(call, slot) for call in calls], dim=0)
            return argument(self.call, slot)

        args = [build(i) for i in range(len(self.call.args))]
        kwargs = {key: build(key) for key in self.call.kwargs}
        return args, kwargs


def argument(call: Call, slot: Any) -> Any:
    """``call``'s argument at ``slot``: a position or a keyword."""
    return call.args[slot] if isinstance(slot, int) else call.kwargs[slot]


class Coalescer:
    """Runs the module calls workers are parked on, sharing a call where it can.

    A runtime whose workers share a forward sets one on its interleaver
    (``interleaver.coalescer``).

    Args:
        tokenwise: Whether a call is token-wise: each output row computed from
            its own input row, the arguments left as they were. It is given the
            `Call`, so it can decide on the arguments as well as on the module.
            It must answer from what every process running the block sees alike
            (the module's class, the number of arguments), so that processes
            running a block in step share the same calls. Without one, only
            registered modules are shared.
    """

    def __init__(self, tokenwise: Callable[[Call], bool] = lambda call: False) -> None:
        self.tokenwise = tokenwise
        self.registered: weakref.WeakSet = weakref.WeakSet()

    def register(self, module: torch.nn.Module) -> None:
        """State that every call of ``module`` is token-wise. This is taken as
        given: registering a module that mixes rows lets one worker's values
        reach another's."""
        self.registered.add(module)

    def shares(self, call: Call) -> bool:
        """Whether ``call`` may run together with other workers' calls."""
        return call.envoy._module in self.registered or self.tokenwise(call)

    def run(self, calls: list) -> list:
        """Run ``calls``, each a ``(worker, Call)``; return ``(worker, Outcome)``
        for each, in the order given."""
        outcomes: dict[int, Outcome] = {}
        for members in self.matching(calls):
            group = self.group(members)
            if group is not None and self.shares(group.call):
                shared = self.together(group)
                if shared is not None:
                    for (worker, _), outcome in zip(group.members, shared):
                        outcomes[id(worker)] = outcome
                    continue
            for worker, call in members:
                outcomes[id(worker)] = self.alone(call)
        return [(worker, outcomes[id(worker)]) for worker, _ in calls]

    # ------------------------------------------------------------------ grouping

    def form(self, call: Call) -> Optional[tuple]:
        """The form of ``call``: what two calls must have in common to run as
        one. ``None`` for a call that only ever runs alone.

        For each tensor argument its dtype, device and the shape of one row; for
        an envoy argument its module; a plain value itself. A call with any
        other argument runs alone: nothing tells whether two such objects mean
        the same to the module.
        """

        def describe(value: Any) -> Optional[tuple]:
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    return None
                return ("rows", value.dtype, value.device, tuple(value.shape[1:]))
            if isinstance(value, Envoy):
                # A module of the tree the callee belongs to, so it lives as
                # long as the callee and its id is not reused meanwhile.
                return ("module", id(value._module))
            if isinstance(value, PLAIN):
                return ("value", type(value), value)
            return None

        described = [describe(value) for value in call.args]
        described += [(key, describe(value)) for key, value in sorted(call.kwargs.items())]
        if any(item is None or item[-1] is None for item in described):
            return None
        return tuple(described)

    def matching(self, calls: list) -> list[list]:
        """``calls`` partitioned into lists of calls of one module in one form.
        A call with no form is a list of its own."""
        matched: dict[Any, list] = {}
        for index, member in enumerate(calls):
            form = self.form(member[1])
            key = (id(member[1].envoy._module), form) if form is not None else index
            matched.setdefault(key, []).append(member)
        return list(matched.values())

    def group(self, members: list) -> Optional[Group]:
        """``members`` as one call, or ``None`` if they cannot run as one: a lone
        call, a call with no rows, or callers whose own arguments disagree on
        their row count."""
        first = members[0][1]
        if len(members) < 2 or self.form(first) is None:
            return None
        slots = [*range(len(first.args)), *first.kwargs]
        # A tensor every caller passes the same object for is passed once.
        rows = [
            slot
            for slot in slots
            if isinstance(argument(first, slot), torch.Tensor)
            and any(argument(call, slot) is not argument(first, slot) for _, call in members)
        ]
        sizes = []
        for _, call in members:
            counts = {argument(call, slot).shape[0] for slot in rows}
            if len(counts) != 1 or 0 in counts:
                return None
            sizes.append(counts.pop())
        return Group(members, rows, sizes)

    # ----------------------------------------------------------------- execution

    def together(self, group: Group) -> Optional[list[Outcome]]:
        """One outcome per member from one call of the module on every member's
        rows, or ``None`` if that call failed and each member has to make its
        own.

        One member's arguments may be what the call raised on, so the error is
        given to no one here: each member's own call raises it for that member
        only. A result that cannot be cut into the members' rows is dropped
        the same way.
        """
        args, kwargs = group.arguments()
        try:
            result = group.call.envoy._run(*args, **kwargs)
        except Exception:
            return None
        try:
            return [Outcome(piece) for piece in split(result, group.sizes)]
        except Unsplittable:
            return None

    def alone(self, call: Call) -> Outcome:
        """``call`` run on its caller's arguments only."""
        try:
            return Outcome(call.envoy._run(*call.args, **call.kwargs))
        except Exception as error:
            return Outcome(error=error)

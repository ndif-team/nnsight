"""Targeting specific occurrences of a location within one run.

A traced call can reach the same module more than once — most commonly a
generation loop, where every module is revisited on each decoded step. Each
visit produces a fresh *occurrence* of that module's locations (its
``.input``/``.output``/...). [`Iterations`][nnsight.intervention.iterator.Iterations], reached as ``tracer.iter``,
lets a stretch of trace body bind its reads and writes to a chosen range of
those occurrences:

.. code-block:: python

    with model.generate("...", max_new_tokens=10) as tracer:
        for step in tracer.iter[:3]:
            hidden = model.transformer.h[0].output.save()  # steps 0, 1, 2

The mechanism lives in the interleaver (see
[`nnsight.intervention.interleaver`][nnsight.intervention.interleaver]): each visit to a location is tagged
with its occurrence index, and a worker's request is tagged with the occurrence
it wants — the running [`Mediator`][nnsight.intervention.interleaver.Mediator]'s
``iteration``. Looping over ``tracer.iter`` is what moves that pointer.
"""

from __future__ import annotations

import warnings
from types import CodeType
from typing import TYPE_CHECKING, Iterator

from greenlet import getcurrent

from ..tracing.tracer import Tracer, push_result
from ..tracing.util import Scope, shared_locals
from .interleaver import OutOfOrderError

if TYPE_CHECKING:
    from .interleaver import Mediator


class Iterations(Tracer):
    """Selects which occurrences of a location a stretch of trace body targets.

    Returned by [`iter`][nnsight.intervention.tracer.InterleavingTracer.iter]. A
    module reached repeatedly in one run — once per step of a generation loop,
    say — produces a fresh occurrence of each of its locations every step.
    Looping over ``tracer.iter`` walks the running
    [`Mediator`][nnsight.intervention.interleaver.Mediator]'s ``iteration`` pointer
    across a range of those occurrences, so reads and writes inside the loop body
    bind to the matching step:

    .. code-block:: python

        with model.generate("...", max_new_tokens=10) as tracer:
            for step in tracer.iter[:3]:
                hidden = model.transformer.h[0].output.save()  # steps 0, 1, 2

    The occurrences are chosen by indexing:

    * a slice — ``tracer.iter[:3]`` is steps 0–2, ``tracer.iter[2:5]`` is 2–4;
    * an int — ``tracer.iter[2]`` is just step 2;
    * a list — ``tracer.iter[[0, 2, 4]]`` is those steps only (the skipped
      occurrences still advance the model's per-location count, so index 2 binds
      the third occurrence regardless of which steps were selected).

    An open end — ``tracer.iter[:]`` — runs until the model stops producing
    steps: the loop keeps handing out step indices, and the final request (for a
    step the model never runs, e.g. one past the last generated token) is simply
    left parked. The interleaver reports that dangling request as a warning rather
    than an error (see
    [`check_dangling_mediators`][nnsight.intervention.interleaver.Interleaver.check_dangling_mediators]),
    keeping the values from every step that did run.

    A ``with tracer.iter[...]:`` block does the same thing and is deprecated; the
    loop moves inside `execute`, which re-runs the block per step. Prefer the
    ``for`` form.

    Attributes:
        start: First occurrence index for a range; ignored when ``steps`` is set.
        stop: One past the last for a range, or ``None`` to run open-ended.
        steps: An explicit list of occurrence indices, or ``None`` for a range.
    """

    def __init__(
        self,
        start: int = 0,
        stop: int | None = None,
        steps: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.steps = steps

    def __getitem__(self, key: slice | int | list[int]) -> Iterations:
        """Select occurrences via ``[:n]`` / ``[a:b]`` / ``[i]`` / ``[[i, j, ...]]``."""
        if isinstance(key, slice):
            return Iterations(key.start or 0, key.stop)
        if isinstance(key, (list, tuple)):
            return Iterations(steps=list(key))
        return Iterations(key, key + 1)

    def _indices(self) -> Iterator[int]:
        """The step indices this selection yields — an explicit list or a range."""
        if self.steps is not None:
            yield from self.steps
            return
        step = self.start
        while self.stop is None or step < self.stop:
            yield step
            step += 1

    def __iter__(self) -> Iterator[int]:
        """Walk the running mediator's ``iteration`` across the selected steps.

        Before yielding each step, pin the mediator's ``iteration`` so the first
        request in the loop body binds to that occurrence (it then relaxes; see
        [`iteration`][nnsight.intervention.interleaver.Mediator.iteration]). An open end
        (``stop is None``) keeps handing out steps indefinitely — the model
        stopping is what ends the loop, via a dangling final request. Whatever
        ``iteration`` was before the loop is restored on exit, so loops can nest.
        """
        # The loop body runs inside the worker greenlet, which carries a weakref
        # to its own mediator (set in Mediator.start).
        mediator: Mediator = getcurrent().mediator()
        previous = mediator.iteration
        try:
            for step in self._indices():
                if step < 0:
                    raise ValueError(f"tracer.iter step cannot be negative: {step}")
                mediator.iteration = step
                yield step
                step += 1
        finally:
            mediator.iteration = previous

    def execute(self, code: CodeType) -> None:
        """Run the block once per selected occurrence — the deprecated ``with`` form.

        ``with tracer.iter[...]:`` predates ``for step in tracer.iter[...]:`` and
        does the same thing the long way: rather than the loop being the user's,
        it is here, re-running the captured block with the mediator's ``iteration``
        pinned to each step (see `__iter__` for the same pinning). Prefer the
        ``for`` form, which is a plain loop over the body written inline.

        Open-ended (``tracer.iter[:]``) ends the way the ``for`` form does: the
        step past the model's last parks, and the interleaver throws
        [`OutOfOrderError`][nnsight.intervention.interleaver.OutOfOrderError] into it once the
        run finishes — caught here, so the reached steps' saved values are kept.
        """
        warnings.warn(
            "`with tracer.iter[...]:` is deprecated; use "
            "`for step in tracer.iter[...]:` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        frame = self.info.frame
        mediator: Mediator = getcurrent().mediator()
        previous = mediator.iteration
        scope = Scope(dict(frame.f_locals), shared_locals(frame), frame.f_globals)
        try:
            for step in self._indices():
                if step < 0:
                    raise ValueError(f"tracer.iter step cannot be negative: {step}")
                mediator.iteration = step
                exec(code, scope)
        except OutOfOrderError:
            pass  # ran fewer steps than asked; keep what the reached steps saved
        finally:
            mediator.iteration = previous
        push_result(frame, scope)

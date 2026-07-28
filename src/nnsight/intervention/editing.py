"""Editing: interventions that stick.

A normal trace runs the model once and forgets your interventions. *Editing*
lets you record a block of interventions once and have it replayed automatically
on every future trace of that envoy — no need to repeat it each time. Think of an
edit as a permanent patch to the model's behavior.

By default [`Envoy.edit`][nnsight.intervention.envoy.Envoy.edit] leaves the original envoy clean and returns an
edited copy, so you opt into the edited behavior through the copy that
``with model.edit() as (tracer, edited):`` binds. Pass ``inplace=True`` to edit
the envoy itself, in which case only the tracer is bound. Stored edits can be
removed with [`Envoy.clear_edits`][nnsight.intervention.envoy.Envoy.clear_edits].

Example:
    >>> import torch.nn as nn
    >>> from nnsight.intervention.envoy import Envoy
    >>>
    >>> model = Envoy(some_module)
    >>>
    >>> # Store an edit; ``model`` itself is left untouched.
    >>> with model.edit() as (tracer, edited):
    ...     edited.layer1.output[:] = 0
    >>>
    >>> # The stored edit is replayed on every trace of ``edited``.
    >>> with edited.trace(x):
    ...     out = edited.output.save()   # reflects layer1.output == 0
    >>> print(out)
    >>>
    >>> # ``model`` still behaves normally.
    >>> with model.trace(x):
    ...     clean = model.output.save()  # unedited
    >>> print(clean)
"""
from __future__ import annotations

from types import CodeType
from typing import TYPE_CHECKING, Any

from ..tracing.backend import Backend
from ..tracing.tracer import skip_context, skippable
from .interleaver import Mediator
from .tracer import InterleavingTracer

if TYPE_CHECKING:
    from .envoy import Envoy


class EditingTracer(InterleavingTracer):
    """Record a block of interventions and replay it on every future trace.

    Returned by [`Envoy.edit`][nnsight.intervention.envoy.Envoy.edit]. An edit block is written like a
    ``with envoy.trace():`` block, but instead of running once it is stored and
    replayed automatically on every later trace of the envoy. With
    ``inplace=False`` (default) the original envoy is left clean and an edited copy
    is returned alongside the tracer; ``inplace=True`` edits the envoy itself and
    binds only the tracer. Edits stack and persist until [`Envoy.clear_edits`][nnsight.intervention.envoy.Envoy.clear_edits].

    Being a trace block, it carries the tracer's ``iter`` API — use it to re-apply
    an edit at every occurrence of a location, e.g. each generation step.

    Examples:
        >>> with model.edit() as (tracer, edited):
        ...     edited.layer1.output[:] = 0
        >>> with edited.trace(x):
        ...     out = edited.output.save()   # reflects the stored edit
        >>> print(out)
    """

    def __init__(
        self,
        envoy: Envoy,
        *,
        inplace: bool = False,
        backend: Backend | None = None,
    ) -> None:
        """Args:
            envoy: The envoy whose behavior the edit will patch.
            inplace: If ``False`` (default), edit a shallow copy so the original
                envoy is left untouched; if ``True``, store the edit on ``envoy``
                itself.
            backend: Optional backend for the underlying trace.
        """
        if not inplace:
            envoy = envoy._shallow_copy()
        super().__init__(envoy, "__call__", backend=backend)
        self.inplace = inplace

    def __enter__(self) -> EditingTracer | tuple[EditingTracer, Envoy]:
        """Enter the edit block.

        Returns the tracer — an edit block is a trace block, so it has the same
        ``tracer.iter`` / ``tracer.all()`` API, needed to re-apply an edit at every
        occurrence of a location (e.g. each step of a generation loop)::

            with model.edit(inplace=True) as tracer:
                for _ in tracer.iter[:]:
                    model.h[0].output[0][:] = model.h[0].adapter(
                        model.h[0].output[0], hook=True
                    )

        With ``inplace=False`` (the default) the edit is stored on a copy, so the
        copy is returned alongside the tracer — bind both and write the block's
        interventions against the copy::

            with model.edit() as (tracer, edited):
                edited.h[0].output[0][:] = 0
        """
        # Mirror Tracer.__enter__ (capture + arm skip) *directly* — not via super()
        # — so capture/skip see the user's frame at the same depth.
        self.capture()
        if skippable(self.node):
            skip_context(self)
        return self if self.inplace else (self, self.envoy)

    def execute(self, code: CodeType) -> None:
        """Store the block instead of running it.

        Overrides the base trace behavior: rather than executing ``code`` against a
        live model run, append it to ``envoy._edits`` so [`Envoy.interleave`][nnsight.intervention.envoy.Envoy.interleave]
        replays it on every later trace.
        """
        # Store the captured block as a mediator (code + scope), don't run it.
        # copy=True so each replay execs against a fresh copy of the scope —
        # repeated traces don't accumulate the block's local mutations.
        self.envoy._edits.append(
            Mediator(
                code,
                self.info.frame.f_globals,
                dict(self.info.frame.f_locals),
                copy=True,
                node=self.node,
            )
        )

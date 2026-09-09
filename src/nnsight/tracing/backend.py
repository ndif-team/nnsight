"""What "run the captured block" means.

A [`Tracer`][nnsight.tracing.tracer.Tracer] captures a ``with`` block but does not
decide where it runs — it hands the block to a **backend**. Swapping the backend
is how the same trace becomes a local run, a remote job, or a simulation without
the tracer knowing the difference. The base backend runs the block in place;
higher layers subclass it to send the block elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracer import Tracer


class Backend:
    """Run a tracer's captured block; the base default runs it in place.

    Subclasses override `__call__` to run the block somewhere else (a remote
    worker, a simulation), reading the compiled code and captured frame from
    [`info`][nnsight.tracing.tracer.Tracer.info].
    """

    def __call__(self, tracer: Tracer) -> None:
        """Execute ``tracer``'s captured block in the caller's process."""
        tracer.execute(tracer.info.code)

"""Carry a deferred worker exception back to the trace that wrote it.

A deferring interleaver ([`defer_exceptions`][nnsight.intervention.interleaver.Interleaver.defer_exceptions])
records a worker's exception on its mediator instead of raising it out of the
hook, so the driver can end just that worker's request. The error still has to
reach the client that wrote the intervention — which may be another process — so
it is reduced to a plain, picklable dict in the worker and re-raised at the client.

What the client catches is a ``RuntimeError`` whose message *starts* with the
original type name and message, so a caller that needs to tell one refusal from
another matches on the text rather than on the class.

A ``tracer.stop()`` raises [`EarlyStopException`][nnsight.intervention.interleaver.EarlyStopException],
which travels the same path but is control flow, not an error: it is marked and
never re-raised.
"""

from __future__ import annotations

import traceback as _traceback
from typing import Optional, TypedDict

from .interleaver import EarlyStopException


class DeferredError(TypedDict, total=False):
    type_name: str
    message: str
    traceback: str
    is_control_flow: bool


def capture_exception(exception: BaseException) -> DeferredError:
    """Reduce a worker's exception to a wire-safe dict.

    Uses the intervention-only traceback stashed by
    [`switch`][nnsight.intervention.interleaver.Mediator.switch] when present, so the
    surfaced trace points at the user's line rather than the model/hook stack.
    """
    tb = getattr(exception, "__intervention_tb__", None) or exception.__traceback__
    return {
        "type_name": type(exception).__name__,
        "message": str(exception),
        "traceback": "".join(_traceback.format_tb(tb)),
        "is_control_flow": isinstance(exception, EarlyStopException),
    }


def raise_deferred(error: Optional[DeferredError]) -> None:
    """Re-raise a captured worker error at the client; a stop returns silently.

    Raised as a ``RuntimeError`` carrying the original type, message, and
    intervention traceback rather than reconstructing the original class, which is
    brittle across a process boundary.
    """
    if not error or error.get("is_control_flow"):
        return
    lines = [f"{error.get('type_name', 'Exception')}: {error.get('message', '')}"]
    traceback_str = error.get("traceback")
    if traceback_str:
        lines += ["", "Intervention traceback:", traceback_str.rstrip()]
    raise RuntimeError("\n".join(lines))

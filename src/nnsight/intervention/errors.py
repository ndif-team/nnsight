"""Shared error-surfacing helpers for server↔client deferred-exception flow.

Defer mode (``Interleaver.defer_exceptions=True``, set permanently on the
vLLM interleaver in ``GPUModelRunner.load_model``) means user-code
exceptions do not escape the hook in server deployments. Someone
downstream must read ``mediator.deferred_exception``, ship the error
back to the right client, and raise it at the client's trace boundary.

This module is the shared piece that makes that raise correct and
consistent across HF-serve, vLLM-serve, and any future server path.
It owns two things:

1. ``capture_deferred(mediator)`` — serialize a deferred exception into
   a plain dict (``DeferredError`` below) that can cross any wire format.
   Preserves the original type name, string message, formatted
   traceback, and whether it's an intentional control-flow signal
   (``EarlyStopException``).

2. ``surface_server_errors(errors, context)`` — filter control-flow
   entries, and if any real errors remain, raise ``RuntimeError`` with
   a message that preserves type/message/traceback for debugging.
   Always raises ``RuntimeError`` rather than reconstructing the
   original exception class — the server boundary is a semantic
   boundary, and faking type identity across it is brittle.
"""

from __future__ import annotations

import logging
import traceback as _tb
from typing import Any, List, Optional, TypedDict


class DeferredError(TypedDict, total=False):
    req_id: str
    type_name: str
    message: str
    traceback: str
    is_control_flow: bool


def capture_deferred(mediator: Any, req_id: Optional[str] = None) -> Optional[DeferredError]:
    """Serialize a mediator's deferred exception into a wire-safe dict.

    Returns None if the mediator has no deferred exception. The caller
    is responsible for clearing ``mediator.deferred_exception`` (and the
    associated ``_deferred_*`` metadata fields) after capture.
    """
    exc = getattr(mediator, "deferred_exception", None)
    if exc is None:
        return None

    # Prefer metadata captured at deferral time (which sees the ORIGINAL
    # exception, before `wrap_exception` replaces the class with a dynamic
    # NNsightException subclass). Fall back to reading from the wrapped
    # exception if the capture-time fields weren't populated.
    type_name = getattr(mediator, "_deferred_type_name", None)
    if type_name is None:
        # Wrapped exceptions are `class NNsightException(original_type, ExceptionWrapper)`.
        # Bases[0] recovers the original. Robust fallback for anything else.
        bases = getattr(type(exc), "__bases__", ())
        type_name = bases[0].__name__ if bases else type(exc).__name__

    traceback_str = getattr(mediator, "_deferred_traceback", None) or ""
    is_control_flow = getattr(mediator, "_deferred_is_control_flow", False)

    entry: DeferredError = {
        "type_name": type_name,
        "message": str(exc),
        "traceback": traceback_str,
        "is_control_flow": bool(is_control_flow),
    }
    if req_id is not None:
        entry["req_id"] = req_id
    return entry


def surface_server_errors(
    errors: Optional[List[DeferredError]],
    context: str = "",
) -> None:
    """Raise the first real deferred error across a server boundary.

    Control-flow entries (``is_control_flow=True``, e.g. ``EarlyStopException``
    from ``tracer.stop()``) are filtered out. If no real errors remain, this
    returns silently.

    Otherwise, raises ``RuntimeError`` whose message includes:
    - ``context`` prefix (e.g. ``"[nnsight-serve]"``)
    - original type name and message
    - the intervention's traceback as a formatted string (when captured)

    The raise type is always ``RuntimeError`` — the server boundary is a
    semantic boundary (different process, potentially different Python
    version, sandboxed imports). Reconstructing the original exception
    class via builtin lookup is fragile and silently degrades when the
    class isn't importable. Keeping everything inside the ``RuntimeError``
    message preserves diagnostic information without the failure mode.
    """
    if not errors:
        return

    real = [e for e in errors if not e.get("is_control_flow", False)]
    if not real:
        return

    first = real[0]
    type_name = first.get("type_name", "Exception")
    message = first.get("message", "")
    traceback_str = first.get("traceback", "")
    req_id = first.get("req_id")

    prefix = f"{context} " if context else ""
    req_suffix = f" (req_id={req_id})" if req_id else ""

    lines = [f"{prefix}{type_name}: {message}{req_suffix}"]
    if traceback_str:
        lines.append("")
        lines.append("Intervention traceback:")
        lines.append(traceback_str.rstrip())
    raise RuntimeError("\n".join(lines))


def log_invalid_payload(
    logger: logging.Logger,
    request_id: str,
    exc: BaseException,
) -> str:
    """Log a deserialization failure server-side and return a generic
    400-detail string safe to send to the client.

    Pickle deserialization errors leak server internals when echoed
    verbatim. ``str(pickle.UnpicklingError)`` typically embeds the
    failing byte offset and module path; ``AttributeError`` from a
    custom ``__reduce__`` chain leaks installed module names; even an
    EOFError can hint at the parser's depth.

    A malicious client can probe by submitting deliberately malformed
    bytes and reading the response — this turns the deserialize path
    into an information-disclosure oracle. Mitigation: log the full
    traceback server-side (operator can grep by ``request_id``), but
    return only the generic message + ``request_id`` to the client.

    The ``request_id`` is generated by the caller (so the same id can
    be threaded through the success-path log entry and any subsequent
    error logs from the same request).
    """
    logger.exception(
        "Invalid request payload (request_id=%s, exception_class=%s)",
        request_id,
        type(exc).__name__,
    )
    return f"Invalid request payload (request_id={request_id})"

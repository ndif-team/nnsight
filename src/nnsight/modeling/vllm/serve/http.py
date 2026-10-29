"""What both halves of the serve client put on the wire.

A trace and an installed block reach a server the same way — the same headers,
the same timeouts, the same reading of a non-200 — but from two unrelated places:
[`LocalServeBackend`][nnsight.modeling.vllm.serve.backend.LocalServeBackend] sends
the trace, and
[`ServeRegistration`][nnsight.modeling.vllm.registration.ServeRegistration] sends
the block. Keeping the wire details here is what stops the two from drifting; the
error message in particular is one a user reads, and two copies of it would
diverge on the first edit.
"""

from __future__ import annotations

from typing import Any, Optional

CONNECT_TIMEOUT: float = 10.0
# Generous: the request is queued behind whatever else the engine is running.
READ_TIMEOUT: float = 600.0


def timeout() -> Any:
    import httpx

    return httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)


def headers(api_key: Optional[str] = None, compress: bool = False) -> dict:
    """The headers every nnsight-serve request carries."""
    sending = {
        "Content-Type": "application/octet-stream",
        "nnsight-compress": str(compress),
    }
    if api_key:
        sending["ndif-api-key"] = api_key
    return sending


def check(response: Any) -> None:
    """Raise the server's own explanation of a non-200, as a `ConnectionError`.

    A transport or service failure, as distinct from an error *inside* the trace —
    that one comes back in the body as a deferred error and is re-raised at the
    client with its real type.
    """
    if response.status_code == 200:
        return
    try:
        detail = response.json().get("detail", response.reason_phrase)
    except Exception:
        detail = response.reason_phrase
    raise ConnectionError(f"nnsight-serve returned {response.status_code}: {detail}")

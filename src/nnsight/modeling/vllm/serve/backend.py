"""Run a trace on a remote nnsight-serve engine.

The backend for ``model.trace(..., serve=url)``. The trace is written against a
meta model — no GPU, no weights — serialized in the same form the remote (NDIF)
path uses, and sent to a running nnsight-serve instance. The server runs it and
returns the saved values, which are pushed back into the caller's frame exactly as
a local run would, so reading a ``.save()``d value after the block just works::

    model = VLLM("gpt2")                                       # no GPU needed
    with model.trace("Hello", serve="http://host:8000", api_key="..."):
        logits = model.logits.save()
    logits                                                     # pushed back here

``api_key`` (optional) is sent as the ``ndif-api-key`` header. The server's response
is a ``torch.save`` of ``{"saves": {name: value}, "error": <deferred or None>}`` —
saved values only; generated tokens are not returned. A build or runtime error comes
back as ``error`` and is re-raised at the client with its real type and traceback; a
transport/service failure (e.g. the engine not ready) surfaces as ``ConnectionError``.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Optional

import torch

from ....tracing.backend import Backend

if TYPE_CHECKING:
    from ....intervention.tracer import Tracer


class LocalServeBackend(Backend):
    """Send a trace to an nnsight-serve instance and bring its saves home."""

    CONNECT_TIMEOUT: float = 10.0
    READ_TIMEOUT: float = 600.0

    def __init__(self, model: Any, host: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.api_key = api_key

    def __call__(self, tracer: "Tracer") -> None:
        from ....schema.request import RequestModel

        blob = RequestModel.serialize(tracer, compress=False)
        saves = self._send(blob)
        self._push(saves, tracer)

    def _send(self, blob: bytes) -> dict:
        import httpx

        from ....intervention.errors import raise_deferred

        headers = {
            "Content-Type": "application/octet-stream",
            "nnsight-compress": "False",
        }
        if self.api_key:
            headers["ndif-api-key"] = self.api_key

        with httpx.Client(
            timeout=httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)
        ) as client:
            response = client.post(
                f"{self.host}/v1/nnsight/generate", content=blob, headers=headers
            )

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.reason_phrase)
            except Exception:
                detail = response.reason_phrase
            raise ConnectionError(
                f"nnsight-serve returned {response.status_code}: {detail}"
            )

        result = torch.load(
            io.BytesIO(response.content), map_location="cpu", weights_only=False
        )
        # A real intervention error on the server comes back as a deferred error;
        # re-raise it here (a tracer.stop() is control flow and stays silent).
        raise_deferred(result.get("error"))
        return result.get("saves", {})

    def _push(self, saves: dict, tracer: "Tracer") -> None:
        from ....tracing.tracer import mark
        from ....tracing.util import push

        # Mark each value saved so the trace's own push_result keeps it, then write
        # the names into the caller's frame — the same path a local run takes. mark,
        # not save: results are pushed back after the trace has exited.
        for value in saves.values():
            mark(value)
        if tracer.info.frame is not None:
            push(tracer.info.frame, saves)

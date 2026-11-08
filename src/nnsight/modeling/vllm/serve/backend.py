"""Run a trace on a remote nnsight-serve engine.

The backend for ``model.trace(..., serve=url)``. The trace is written against a
meta model (no GPU, no weights). This backend builds the trace's requests here, one
per invoke with that invoke's serialized block in its ``SamplingParams``, the same
requests an in-process engine is given, and sends them to a running nnsight-serve
instance. The server submits them to its engine and returns the saved values, which
are pushed back into the caller's frame exactly as a local run would, so reading a
``.save()``d value after the block just works::

    model = VLLM("gpt2")                                       # no GPU needed
    with model.trace("Hello", serve="http://host:8000", api_key="..."):
        logits = model.logits.save()
    logits                                                     # pushed back here

``api_key`` (optional) is sent as the ``ndif-api-key`` header. The server's response
is a ``torch.save`` of ``{"saves": {name: value}, "error": <deferred or None>}`` —
saved values only, so read generated tokens by saving ``tracer.result``. An error
building the requests raises here, before anything is sent. A runtime error comes
back as ``error`` and is re-raised at the client with its real type and traceback; a
transport/service failure (e.g. the engine not ready) surfaces as ``ConnectionError``.
"""

from __future__ import annotations

import io
import pickle
from typing import TYPE_CHECKING, Any, Optional

import torch

from ....tracing.backend import Backend

if TYPE_CHECKING:
    from ....intervention.tracer import Tracer


class LocalServeBackend(Backend):
    """Send a trace to an nnsight-serve instance and bring its saves home."""

    def __init__(self, model: Any, host: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.api_key = api_key

    def __call__(self, tracer: "Tracer") -> None:
        from ....intervention.util import clear_shared_locals

        model = self.model
        try:
            # What `VLLM._call` does before it hands the requests to its engine;
            # the meta model has none, so the server's engine takes them.
            _, (prompts, params, lora_requests), forward_kwargs = tracer.prepare(
                tracer.info.code
            )
            model._attach_mediators(params, **forward_kwargs)
        finally:
            # The workers are serialized onto the requests now; the live ones are done.
            model.interleaver.cancel()
            clear_shared_locals()
        saves = self._send(pickle.dumps((prompts, params, lora_requests)))
        self._push(saves, tracer)

    def _send(self, blob: bytes) -> dict:
        import httpx

        from ....intervention.errors import raise_deferred
        from . import http

        with httpx.Client(timeout=http.timeout()) as client:
            response = client.post(
                f"{self.host}/v1/nnsight/generate",
                content=blob,
                headers=http.headers(self.api_key),
            )
        http.check(response)

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

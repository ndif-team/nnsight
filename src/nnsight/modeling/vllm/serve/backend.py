"""Run a trace on a remote nnsight-serve engine.

The backend for ``model.trace(..., serve=url)``. The client holds a meta-device
[`VLLM`][nnsight.modeling.vllm.vllm.VLLM] model (no GPU, no weights). On trace exit,
[`LocalServeBackend`][nnsight.modeling.vllm.serve.backend.LocalServeBackend] runs
``tracer.prepare`` and ``model._attach_mediators`` to build one request per invoke,
with each invoke's [`Mediator`][nnsight.intervention.interleaver.Mediator] serialized
into ``SamplingParams.extra_args["nnsight_mediator"]``, then pickles
``(prompts, params, lora_requests)`` and POSTs the payload to ``/v1/nnsight/generate``
on a running nnsight-serve instance. The server runs the requests on its engine and
returns the saved values, which are written into the caller's frame::

    model = VLLM("gpt2")                                       # no GPU needed
    with model.trace("Hello", serve="http://host:8000", api_key="..."):
        logits = model.logits.save()
    logits                                                     # pushed back here

``api_key`` (optional) is sent as the ``ndif-api-key`` header. The server's response
is a ``torch.save`` payload of ``{"saves": {name: value}, "error": <deferred or None>}``.
Only saved values are returned, so read generated tokens by saving ``tracer.result``.
Errors during ``tracer.prepare`` or ``_attach_mediators`` (such as an invoke without a
prompt or an unknown sampling argument) raise in the client before the request is sent.
An exception raised while an intervention runs on the server is returned in ``error``
and re-raised at the client by [`raise_deferred`][nnsight.intervention.errors.raise_deferred]
with its original type name and traceback. A non-200 response (such as a 503 when
the engine is not ready) raises ``ConnectionError``, and a server that cannot be
reached raises ``httpx.ConnectError``.
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
            # Collect each invoke's prompt, SamplingParams, and Mediator on the client
            # meta model, then serialize each Mediator into SamplingParams.extra_args,
            # matching the preparation VLLMTracer.execute and VLLM._call run locally.
            _, (prompts, params, lora_requests), forward_kwargs = tracer.prepare(
                tracer.info.code
            )
            model._attach_mediators(params, **forward_kwargs)
        finally:
            # LocalServeBackend runs in place of VLLMTracer.execute, so clear the
            # client's interleaver and shared-locals store here once the Mediators
            # have been serialized onto `params`.
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

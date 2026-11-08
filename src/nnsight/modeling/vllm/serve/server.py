"""A local HTTP server that runs nnsight traces on a vLLM engine.

The server holds one dispatched async [`VLLM`][nnsight.modeling.vllm.vllm.VLLM].
A client with only a meta model (no GPU) writes a trace as usual and sends it here
with ``model.trace(..., serve=url)``. The client builds the trace's requests, one
per invoke with that invoke's serialized block in ``SamplingParams.extra_args``, the
same requests an in-process engine is given; this server submits them to its engine,
collects the saved values, and returns them. The blocks reach the engine's workers
unopened: this process unpickles only the prompts and sampling settings, and never
rebuilds a tracer.
"""

from __future__ import annotations

import asyncio
import io
import logging
import pickle
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response

from ..engines.engine import acollect

if TYPE_CHECKING:
    from ..vllm import VLLM

logger = logging.getLogger("nnsight.serve")

app = FastAPI(title="nnsight-vllm-serve")

# Set by the CLI once the engine is up.
_model: Optional["VLLM"] = None

# Strong references to fire-and-forget worker-cleanup tasks, so the loop doesn't
# garbage-collect them mid-flight (asyncio tracks tasks weakly).
_cleanups: set = set()


def _payload(saves: dict, error: Optional[dict]) -> Response:
    """A generate response: torch.save of ``{"saves": {...}, "error": <deferred>}``."""
    buffer = io.BytesIO()
    torch.save({"saves": saves, "error": error}, buffer)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


def set_model(model: "VLLM") -> None:
    """Register the dispatched engine the server runs traces on."""
    global _model
    _model = model


def _ready() -> "VLLM":
    """The dispatched engine, or a 503 — what every endpoint needs before it starts.

    Both halves matter: `set_model` runs before the engine is built, so a request
    arriving in between finds a model whose ``vllm_entrypoint`` is still None.
    """
    if _model is None or _model.vllm_entrypoint is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return _model


@app.get("/health")
async def health() -> dict:
    _ready()
    return {"status": "ok"}


@app.post("/v1/nnsight/register/{registration_id}")
async def register(
    registration_id: str, request: Request, name: Optional[str] = None
) -> dict:
    """Install a block on this server's engine, to run for every request it handles.

    Body: the serialized block ``model.edit(serve=url)`` prepared on the client.
    The client has no engine to ``collective_rpc`` into, so the install comes
    over HTTP and this hands it to every rank — the same call the in-process form
    makes. What the block saves rides the output of the request it ran on, so a
    serve client reads it through ``tracer.result``. ``?name=`` is the name
    requests may address the edit by (``trace(..., edits=[name])``); without it
    the edit runs for every request.
    """
    model = _ready()

    # The client cannot check this — it has no engine — so the server does.
    from ..registration import _warn_if_prefix_caching

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_prefix_caching(model)
    for warning in caught:
        logger.warning(str(warning.message))

    payload = await request.body()
    try:
        await model.vllm_entrypoint.collective_rpc(
            "nnsight_register", args=(registration_id, payload, name)
        )
    except Exception as exception:
        logger.exception("Failed to install edit %s", registration_id)
        raise HTTPException(status_code=400, detail=str(exception))
    return {"status": "ok", "id": registration_id, "name": name}


@app.post("/v1/nnsight/register/{registration_id}/clear")
async def clear_registration(registration_id: str) -> dict:
    """Stop running an installed block. Unknown ids are a no-op, so clearing twice is safe."""
    await _ready().vllm_entrypoint.collective_rpc(
        "nnsight_clear_registered", args=(registration_id,)
    )
    return {"status": "ok", "id": registration_id}


@app.post("/v1/nnsight/generate")
async def generate(request: Request) -> Response:
    """Run a trace's requests and return their saved values.

    Body: a pickle of ``(prompts, params, lora_requests)``, one entry per invoke, as
    [`LocalServeBackend`][nnsight.modeling.vllm.serve.backend.LocalServeBackend]
    builds them; each invoke's block rides its ``SamplingParams.extra_args`` to the
    workers unopened. Response: ``torch.save`` of
    ``{"saves": {name: value}, "error": <deferred error or None>}`` — saved values
    only, so read generated tokens by saving ``tracer.result``, which is served here
    like anywhere else. A body that does not unpickle or a runtime intervention error
    both come back as the deferred ``error`` so the client re-raises the real
    exception type with its traceback, rather than an opaque HTTP error.
    """
    from ....intervention.errors import capture_exception

    model = _ready()

    try:
        prompts, params, lora_requests = pickle.loads(await request.body())
    except Exception as exception:
        logger.exception("Failed to read the trace's requests")
        return _payload({}, capture_exception(exception))

    engine = model.vllm_entrypoint

    async def run_invoke(prompt: Any, param: Any, lora_request: Any) -> tuple:
        request_id = str(uuid.uuid4())
        collected = False
        try:
            # The last streamed output is the finished ``RequestOutput``; the
            # worker cannot build it and needs it to serve ``tracer.result``, so
            # keep it and hand it back on the collect.
            output = None
            async for output in engine.generate(
                prompt, param, request_id, lora_request=lora_request
            ):
                pass

            entry = await acollect(engine, request_id, output)
            collected = True
            if entry is not None:
                return entry["saves"], entry["error"]
            return {}, None
        finally:
            # If this task is cancelled mid-generation (server shutdown, or a
            # disconnect the server is configured to propagate), the request is never
            # collected and its worker would leak. Free it best-effort: fire the collect
            # as a task kept alive by `_cleanups`, and shield the await from the
            # cancellation. Stock uvicorn does not cancel a handler on client
            # disconnect, so under it this path simply does not run.
            if not collected:
                cleanup = asyncio.ensure_future(acollect(engine, request_id))
                _cleanups.add(cleanup)
                cleanup.add_done_callback(_cleanups.discard)
                await asyncio.shield(cleanup)

    tasks = [
        run_invoke(prompt, param, lora_requests[index] if lora_requests else None)
        for index, (prompt, param) in enumerate(zip(prompts, params))
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge each invoke's saves (later invoke wins a name clash, matching the local
    # push). A generation failure surfaces as the deferred error, like a build error.
    all_saves: dict = {}
    first_error = None
    for result in results:
        if isinstance(result, BaseException):
            logger.exception("Generation failed", exc_info=result)
            if first_error is None:
                first_error = capture_exception(result)
            continue
        saves, error = result
        all_saves.update(saves)
        if error is not None and first_error is None:
            first_error = error

    return _payload(all_saves, first_error)

"""A local HTTP server that runs nnsight traces on a vLLM engine.

The server holds one dispatched async :class:`~nnsight.modeling.vllm.vllm.VLLM`.
A client with only a meta model (no GPU) writes a trace as usual and sends it here
with ``model.trace(..., serve=url)``; the server deserializes it, builds and submits
the request to its engine, collects the saved values, and returns them. The trace
is authored on the client and *run* on the server — the same serialized-trace format
the remote (NDIF) path uses, so the client needs no weights.

The request handler builds its workers synchronously (deserialize → prepare →
attach), all before the first ``await``, so concurrent requests never interleave
while sharing the model's one interleaver.
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import pickle
import uuid
from typing import TYPE_CHECKING, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response

from ....intervention.serialization import loads

if TYPE_CHECKING:
    from ..vllm import VLLM

logger = logging.getLogger("nnsight.serve")

app = FastAPI(title="nnsight-vllm-serve")

# Set by the CLI once the engine is up.
_model: Optional["VLLM"] = None
_persistent_objects: Optional[dict] = None

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
    global _model, _persistent_objects
    _model = model
    _persistent_objects = model._remoteable_persistent_objects()


def _build_tracer(body: bytes, compress: bool) -> Any:
    """Rebuild a runnable tracer from a serialized trace, bound to this model.

    Mirrors :meth:`RequestModel.deserialize` — recompile the block from its source
    onto the tracer — but also restores the tracer's AST node. Deserialize drops it
    (server-side mediators normally just run in place), yet here they are
    re-serialized onto the engine's requests, which needs the AST; parsing the
    shipped source back gives an equivalent node whose ``body`` reduces the same way.
    """
    import linecache

    if compress:
        import zstandard as zstd

        body = zstd.ZstdDecompressor().decompress(body)

    tracer, (source, glbls, variables) = loads(body, _persistent_objects)
    frame = tracer.info.frame
    filename = frame.f_code.co_filename
    tracer.info.code = compile(source, filename, "exec").replace(
        co_name=frame.f_code.co_name
    )
    # `filename` is the client's (e.g. "<stdin>"), so requests from different clients
    # collide on this process-global cache. Safe only because the whole build runs
    # without an await (see the module docstring): no two builds interleave, so the
    # source a traceback resolves is always the one just written. An await added to
    # the build path would break that — key by request id then.
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    frame.f_globals = glbls
    frame.f_locals = variables
    tracer.node = ast.parse(source)
    return tracer


@app.get("/health")
async def health() -> dict:
    if _model is None or _model.vllm_entrypoint is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {"status": "ok"}


@app.post("/v1/nnsight/generate")
async def generate(request: Request) -> Response:
    """Run a serialized trace and return its saved values.

    Body: ``RequestModel.serialize(tracer)`` bytes. Response: ``torch.save`` of
    ``{"saves": {name: value}, "error": <deferred error or None>}`` — saved values
    only; generated tokens are not returned (a serve client reads ``.save()``d values,
    not ``tracer.result``). A build error (a corrupt/incompatible trace) or a runtime
    intervention error both come back as the deferred ``error`` so the client re-raises
    the real exception type with its traceback, rather than an opaque HTTP error.
    """
    from ....intervention.errors import capture_exception

    if _model is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    body = await request.body()
    compress = request.headers.get("nnsight-compress", "False").lower() == "true"

    try:
        # The rebuilt tracer is a VLLMTracer whose interleaver resolves to this
        # server's — so building its workers puts them on this model.
        tracer = _build_tracer(body, compress)
        _, (prompts, params, lora_requests), forward_kwargs = tracer.prepare(
            tracer.info.code
        )
        _model._attach_mediators(params, **forward_kwargs)
    except Exception as exception:
        logger.exception("Failed to build trace")
        return _payload({}, capture_exception(exception))
    finally:
        # prepare appends this request's workers onto the shared interleaver; clear
        # them whether the build succeeded (they are serialized onto the request now)
        # or failed (or they would bleed into the next request's build).
        _model.interleaver.cancel()

    engine = _model.vllm_entrypoint

    async def run_invoke(prompt: Any, param: Any, lora_request: Any) -> tuple:
        request_id = str(uuid.uuid4())
        collected = False
        try:
            async for _ in engine.generate(
                prompt, param, request_id, lora_request=lora_request
            ):
                pass

            results = await engine.collective_rpc(
                "collect_nnsight", args=([request_id], [request_id])
            )
            collected = True
            payload = next((result for result in results if result is not None), None)
            if payload is not None:
                entry = pickle.loads(payload).get(request_id)
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
                cleanup = asyncio.ensure_future(
                    engine.collective_rpc(
                        "collect_nnsight", args=([request_id], [request_id])
                    )
                )
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
        if isinstance(result, Exception):
            logger.exception("Generation failed", exc_info=result)
            if first_error is None:
                first_error = capture_exception(result)
            continue
        saves, error = result
        all_saves.update(saves)
        if error is not None and first_error is None:
            first_error = error

    return _payload(all_saves, first_error)

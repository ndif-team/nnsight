"""nnsight-vllm-serve: local HTTP server for nnsight interventions on vLLM.

Design choices:
- Synchronous request/response: client blocks until generation + saves collection
  completes. No WebSocket or polling needed. This differs from NDIF's async
  job-based API because the server is local and single-model.
- Server-side trace compilation: the client sends a serialized RequestModel
  (same format as NDIF). The server deserializes, compiles the trace, creates
  mediators, and submits to the vLLM engine. This means the client does NOT
  need a GPU or a dispatched model — only a meta model for envoy proxies.
- Binary transport: request and response bodies are serialized with
  CustomCloudPickler / torch.save. No JSON encoding of tensors.
- Single model per server instance: different models run on different ports.
"""

from __future__ import annotations

import io
import logging
import uuid
from typing import TYPE_CHECKING, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response

from ....schema.request import RequestModel
from ..exceptions import (
    EngineNotDispatchedError,
    GenerationError,
    TraceCompilationError,
)
from ..execute import execute_request

if TYPE_CHECKING:
    from ..vllm import VLLM

logger = logging.getLogger("nnsight.serve")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[nnsight-serve] %(message)s"))
    logger.addHandler(_handler)

app = FastAPI(title="nnsight-vllm-serve")

# Set by cli.py after engine initialization.
_model: Optional["VLLM"] = None
_persistent_objects: Optional[dict] = None


def set_model(model: "VLLM") -> None:
    global _model, _persistent_objects
    _model = model
    _persistent_objects = model._remoteable_persistent_objects()


@app.get("/health")
async def health():
    if _model is None or _model.vllm_entrypoint is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {"status": "ok"}


@app.post("/v1/nnsight/generate")
async def generate(request: Request):
    """Execute an nnsight trace on the vLLM engine.

    Accepts a serialized RequestModel (same format as NDIF's POST /request).
    Deserializes, compiles the trace into mediators, submits to the engine,
    collects saves, and returns them synchronously.

    Request body: RequestModel.serialize() bytes (optionally zstd-compressed).
    Headers:
        nnsight-compress: "True" or "False" (default "False")

    Response body: torch.save-serialized dict with keys:
        saves: dict of saved variable names → values
        outputs: list of generation output dicts (text, token_ids, etc.)
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    body = await request.body()
    compress = request.headers.get("nnsight-compress", "False").lower() == "true"

    request_id = str(uuid.uuid4())
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        "Received request: id=%s, %d bytes, compress=%s, client=%s",
        request_id, len(body), compress, client_host,
    )
    try:
        request_model = RequestModel.deserialize(
            body, _persistent_objects, compress
        )
    except Exception as e:
        # Generic 400 to avoid leaking pickle internals; full traceback is
        # in the server log under this request_id. See I2 / errors.py.
        from ....intervention.errors import log_invalid_payload
        raise HTTPException(
            status_code=400,
            detail=log_invalid_payload(logger, request_id, e),
        )

    # Engine readiness check is duplicated against ``execute_request``'s
    # own ``EngineNotDispatchedError`` raise (defense-in-depth): the CLI
    # asserts dispatch at startup, so hitting this branch indicates an
    # inconsistent server state. Auto-dispatch from an HTTP handler is
    # a TOCTOU hazard (two concurrent first-requests can both observe
    # ``dispatched=False`` and race into ``dispatch()``), so we 503
    # rather than silently dispatching.
    if not _model.dispatched:
        raise HTTPException(
            status_code=503,
            detail="Engine not dispatched. Server startup should have "
                   "ensured dispatch before accepting requests.",
        )

    try:
        result = await execute_request(_model, request_model)
    except EngineNotDispatchedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except TraceCompilationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except GenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Serialize response using torch.save (handles tensors correctly).
    # ``result`` already has the {saves, errors, outputs} shape — wire-
    # compatible with pre-refactor responses.
    buf = io.BytesIO()
    torch.save(result, buf)
    resp_bytes = buf.getvalue()

    logger.info(
        "Completed request: %d invokes, %d saves (%s), %d errors, %d bytes response",
        len(result["outputs"]),
        len(result["saves"]),
        ", ".join(result["saves"].keys()) if result["saves"] else "none",
        len(result["errors"]),
        len(resp_bytes),
    )
    return Response(content=resp_bytes, media_type="application/octet-stream")

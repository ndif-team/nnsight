"""nnsight-serve: local HTTP server for nnsight interventions with vanilla batching.

Uses ``LanguageModel`` + ``VanillaBatchServer`` for faithful inference
with continuous batching. No paged attention — internal operations
are identical to ``model.generate()``.

Architecture:
- Synchronous request/response: client blocks until generation + saves completes.
- Server-side trace compilation: client sends a serialized RequestModel.
- Binary transport: request/response are pickled bytes / torch.save.
- Single model per server instance.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response

from ....intervention.backends.base import Backend
from ....intervention.tracing.globals import Globals
from ....schema.request import RequestModel

if TYPE_CHECKING:
    from ...language import LanguageModel
    from ..vanilla_server import VanillaBatchServer

logger = logging.getLogger("nnsight.serve")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[nnsight-serve] %(message)s"))
    logger.addHandler(_handler)

app = FastAPI(title="nnsight-serve")

# Set by cli.py after model initialization.
_model: Optional["LanguageModel"] = None
_server: Optional["VanillaBatchServer"] = None
_persistent_objects: Optional[dict] = None


def set_model(model: "LanguageModel", server: "VanillaBatchServer") -> None:
    global _model, _server, _persistent_objects
    _model = model
    _server = server
    _persistent_objects = model._remoteable_persistent_objects()


@app.get("/health")
async def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {"status": "ok"}


@app.get("/models")
async def models():
    if _model is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {
        "model": _model.repo_id,
        "server_running": _server is not None and _server.is_running(),
    }


@app.post("/v1/nnsight/generate")
async def generate(request: Request):
    """Execute an nnsight trace with vanilla batching.

    Request body: RequestModel.serialize() bytes.
    Response body: torch.save-serialized dict with saves.

    Cross-request batching: the compile-and-extract block is sync so it
    runs atomically on the asyncio event loop (no handler interleaving
    during mediator extraction). Once entries are submitted via
    ``submit_async``, this handler awaits ``asyncio.gather`` — yielding
    control to the event loop so other concurrent handlers can submit
    their requests into the same scheduling window.
    """
    if _model is None or _server is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    body = await request.body()
    compress = request.headers.get("nnsight-compress", "False").lower() == "true"

    client_host = request.client.host if request.client else "unknown"
    logger.info(
        "Received request: %d bytes, compress=%s, client=%s",
        len(body), compress, client_host,
    )

    try:
        request_model = RequestModel.deserialize(
            body, _persistent_objects, compress
        )
    except Exception as e:
        logger.exception("Failed to deserialize request")
        raise HTTPException(status_code=400, detail=f"Deserialization failed: {e}")

    fn = request_model.interventions
    tracer = request_model.tracer

    # --- Sync-atomic compile + extract + submit ---
    # No `await` between _setup_interleaver and submit_async, so two
    # concurrent handlers cannot race on the shared model._interleaver.
    try:
        Globals.enter()
        _args, kwargs = tracer._setup_interleaver(fn)
        entries = _server.build_entries(kwargs)
        tracer.mediators.clear()
        futures = [_server.submit_async(req) for req in entries]
    except Exception as e:
        logger.exception("Failed to compile trace")
        raise HTTPException(status_code=400, detail=f"Trace compilation failed: {e}")
    finally:
        Globals.exit()

    # --- Await all per-invoke futures concurrently ---
    # Yielding here is what allows other handlers to submit their
    # requests, enabling cross-request batching in the scheduler.
    results = await asyncio.gather(*futures)

    # --- Merge saves across invokes ---
    all_saves = {}
    for result in results:
        if result and "__error__" not in result:
            all_saves.update(result)

    response_data = {"saves": all_saves}
    buf = io.BytesIO()
    torch.save(response_data, buf)
    resp_bytes = buf.getvalue()

    logger.info(
        "Completed request: %d invokes, %d saves (%s), %d bytes response",
        len(entries),
        len(all_saves),
        ", ".join(all_saves.keys()) if all_saves else "none",
        len(resp_bytes),
    )
    return Response(content=resp_bytes, media_type="application/octet-stream")


def patch_transformers_serve(model) -> None:
    """Inject NNsightCBManager into a HF model for transformers serve compatibility.

    After calling this, ``model.init_continuous_batching()`` returns an
    ``NNsightCBManager`` instead of the base ``ContinuousBatchingManager``,
    so ``transformers serve`` gets nnsight interventions transparently.

    Args:
        model: A HuggingFace ``PreTrainedModel`` instance (the raw model,
            not an NNsight wrapper). Must have ``config``, ``device``, ``dtype``.
    """
    from ..manager import NNsightCBManager
    from ....modeling.language import LanguageModel
    from ...common.request_helper import NNsightRequestHelper

    # Wrap the raw HF model with LanguageModel for nnsight hooks
    nnsight_model = LanguageModel(model, tokenizer=getattr(model, "tokenizer", None))

    request_helper = NNsightRequestHelper()

    manager = NNsightCBManager(
        model=model,
        generation_config=model.generation_config,
        nnsight_model=nnsight_model,
        request_helper=request_helper,
    )

    # Cache it so init_continuous_batching() returns it
    model._cached_continuous_batching_manager = manager
    logger.info("Patched model for nnsight-aware transformers serve")

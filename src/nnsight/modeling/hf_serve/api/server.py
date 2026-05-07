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
import uuid
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

    fn = request_model.interventions
    tracer = request_model.tracer

    # --- Sync-atomic compile + extract + submit ---
    # No `await` between _run_user_fn and submit_async, so two concurrent
    # handlers cannot race with each other on the compile step. We
    # deliberately do NOT call ``_init_shared_interleaver()``: the
    # background generation thread owns ``model.interleaver``, and
    # calling ``Interleaver.initialize`` here would reset
    # ``interleaver.current=None`` mid-``Mediator.start()``, causing a
    # worker to read None at its first ``.output`` access and crash the
    # whole in-flight batch.
    try:
        _args, kwargs = tracer._run_user_fn(fn)
        entries = _server.build_entries(kwargs, mediators=tracer.mediators)
        tracer.mediators.clear()
        futures = [_server.submit_async(req) for req in entries]
    except Exception as e:
        logger.exception("Failed to compile trace")
        raise HTTPException(status_code=400, detail=f"Trace compilation failed: {e}")

    # --- Await all per-invoke futures concurrently ---
    # Yielding here is what allows other handlers to submit their
    # requests, enabling cross-request batching in the scheduler.
    results = await asyncio.gather(*futures)

    # --- Merge saves + errors across invokes ---
    # Typed envelope: saves from every successful invoke are merged;
    # every failed invoke contributes a DeferredError dict to ``errors``.
    # The client (``local_serve.py``) calls ``surface_server_errors`` on
    # ``errors`` after pushing saves, so partial-success saves are
    # preserved before the raise — matches local-trace semantics.
    all_saves = {}
    errors = []
    for result in results:
        if not result:
            continue
        err = result.pop("__error__", None)
        if err is not None:
            # Legacy string-form payloads (defensive: the bg thread's
            # catch-all also emits dict-form now, but a downgrade path
            # here is cheap and keeps the envelope shape uniform).
            if isinstance(err, str):
                err = {
                    "type_name": "Exception",
                    "message": err,
                    "traceback": "",
                    "is_control_flow": False,
                }
            errors.append(err)
        if result:
            all_saves.update(result)

    response_data = {"saves": all_saves, "errors": errors}
    buf = io.BytesIO()
    torch.save(response_data, buf)
    resp_bytes = buf.getvalue()

    logger.info(
        "Completed request: %d invokes, %d saves (%s), %d errors, %d bytes response",
        len(entries),
        len(all_saves),
        ", ".join(all_saves.keys()) if all_saves else "none",
        len(errors),
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

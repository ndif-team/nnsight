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

import asyncio
import io
import logging
import pickle
import uuid
from typing import TYPE_CHECKING, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response

from ....intervention.backends.base import Backend
from ....intervention.tracing.globals import Globals
from ....schema.request import RequestModel

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

    try:
        Globals.enter()

        # Set up mediators from the compiled intervention function.
        args, kwargs = tracer._setup_interleaver(fn)

        if not _model.dispatched:
            # Should already be dispatched by cli.py, but just in case.
            _model.dispatch()

        # Serialize mediators into SamplingParams.extra_args.
        prompts, params, lora_requests = _model._serialize_mediators(
            *args, **kwargs
        )

        tracer.mediators.clear()
    except Exception as e:
        logger.exception("Failed to compile trace")
        raise HTTPException(status_code=400, detail=f"Trace compilation failed: {e}")
    finally:
        Globals.exit()

    # Submit all invokes concurrently and collect saves.
    engine = _model.vllm_entrypoint
    all_saves = {}
    generation_outputs = []

    async def run_invoke(prompt, param, lora_request):
        request_id = str(uuid.uuid4())
        last_output = None

        async for output in engine.generate(
            prompt, param, request_id, lora_request=lora_request
        ):
            last_output = output

        # Collect saves from workers.
        finished = [request_id]
        results = await engine.collective_rpc(
            "collect_nnsight",
            args=([request_id], finished),
        )
        saves_bytes = next((r for r in results if r is not None), None)
        saves = pickle.loads(saves_bytes) if saves_bytes else {}

        gen_output = {}
        if last_output is not None:
            out = last_output.outputs[0] if last_output.outputs else None
            gen_output = {
                "text": out.text if out else "",
                "token_ids": list(out.token_ids) if out else [],
            }

        return saves, gen_output

    tasks = []
    for i, (prompt, param) in enumerate(zip(prompts, params)):
        lora_req = lora_requests[i] if lora_requests else None
        tasks.append(run_invoke(prompt, param, lora_req))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.exception("Generation failed", exc_info=result)
            raise HTTPException(
                status_code=500, detail=f"Generation failed: {result}"
            )
        saves, gen_output = result
        all_saves.update(saves)
        generation_outputs.append(gen_output)

    # Serialize response using torch.save (handles tensors correctly).
    response_data = {"saves": all_saves, "outputs": generation_outputs}
    buf = io.BytesIO()
    torch.save(response_data, buf)
    resp_bytes = buf.getvalue()

    logger.info(
        "Completed request: %d invokes, %d saves (%s), %d bytes response",
        len(generation_outputs),
        len(all_saves),
        ", ".join(all_saves.keys()) if all_saves else "none",
        len(resp_bytes),
    )
    return Response(content=resp_bytes, media_type="application/octet-stream")

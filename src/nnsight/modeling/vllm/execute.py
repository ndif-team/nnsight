"""In-process execution of a deserialized ``RequestModel``.

This module is the execution core of nnsight-vllm-serve, factored out
of the FastAPI route so the same logic can run from any in-process
caller (NDIF Ray actor, future schedulers, direct tests) without
HTTP loopback or copy-paste.

The standalone server (`serve/server.py`) reduces to a thin shell:
deserialize → call :func:`execute_request` → torch.save → return.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import uuid
from typing import TYPE_CHECKING, Any, List, Tuple, TypedDict

from ...intervention.tracing.globals import Globals
from .exceptions import (
    EngineNotDispatchedError,
    GenerationError,
    TraceCompilationError,
)

if TYPE_CHECKING:
    from ...schema.request import RequestModel
    from .vllm import VLLM

logger = logging.getLogger("nnsight.vllm.execute")


class GenerationOutput(TypedDict):
    """One invoke's generation output."""

    text: str
    token_ids: List[int]


class ExecuteResult(TypedDict):
    """Return shape of :func:`execute_request`.

    Wire-compatible with the FastAPI route's response body: the same
    three keys are torch.saved into the HTTP response, so byte-level
    parity with pre-refactor responses is preserved.
    """

    saves: dict
    errors: List[dict]
    outputs: List[GenerationOutput]


def _compile_invokes(
    model: "VLLM",
    tracer,
    fn,
) -> Tuple[List[Any], List[Any], List[Any], dict]:
    """Run a tracer's intervention function and serialize its mediators.

    Returns ``(prompts, params, lora_requests, kwargs)``. The first three
    are the per-invoke fan-out fed to ``engine.generate``; ``kwargs`` is
    the batched kwargs dict surfaced for callers that store it (the async
    backend keeps it on ``self._kwargs`` for streaming).

    The body must remain synchronous. ``Globals.cache`` is process-wide;
    yielding to the event loop between ``Globals.enter`` and
    ``Globals.exit`` would open a check-while-mutating window for any
    concurrent caller.

    The helper deliberately does **not**:

    - call ``tracer._init_shared_interleaver()`` — the vLLM worker thread
      owns ``model.interleaver`` and a re-init from here races with
      ``Mediator.start()`` (see ``Interleaver.initialize`` for the
      asserted invariant);
    - dispatch the engine — callers decide policy. The server handler
      raises :class:`EngineNotDispatchedError`; ``AsyncVLLMBackend``
      dispatches on the local-trace path before calling in.

    Exceptions are not caught. Callers wrap appropriately for their
    boundary (the server handler raises :class:`TraceCompilationError`;
    the async backend wraps with ``wrap_exception(e, tracer.info)``).
    """
    try:
        Globals.enter()

        args, kwargs = tracer._run_user_fn(fn)

        prompts, params, lora_requests = model._serialize_mediators(
            *args, mediators=tracer.mediators, **kwargs
        )

        tracer.mediators.clear()
        return prompts, params, lora_requests, kwargs
    finally:
        Globals.exit()


async def execute_request(
    model: "VLLM",
    request_model: "RequestModel",
) -> ExecuteResult:
    """Execute a deserialized ``RequestModel`` on a dispatched ``VLLM``.

    Args:
        model: A ``VLLM`` instance with ``dispatched=True``. Callers must
            dispatch beforehand — ``execute_request`` raises
            :class:`EngineNotDispatchedError` rather than auto-dispatching
            (auto-dispatch from a request boundary is a TOCTOU hazard:
            two concurrent first-callers can both observe
            ``dispatched=False`` and race into ``model.dispatch()``).
        request_model: A deserialized ``RequestModel`` with ``.tracer``
            and ``.interventions`` populated. Typically built by
            ``RequestModel.deserialize(body, persistent_objects)`` on
            the server side.

    Returns:
        :class:`ExecuteResult` with merged ``saves``, the per-mediator
        deferred-error envelope as ``errors``, and one
        :class:`GenerationOutput` per invoke.

    Raises:
        EngineNotDispatchedError: ``model`` is not dispatched.
        TraceCompilationError: compiling the user's intervention function
            or serializing mediators failed before any engine work was
            scheduled.
        GenerationError: ``engine.generate`` or
            ``collective_rpc("collect_nnsight", …)`` raised on any invoke
            task. Distinct from per-mediator deferred user-code errors,
            which surface via the ``errors`` field on the result.
    """
    if not model.dispatched:
        raise EngineNotDispatchedError(
            "VLLM model is not dispatched. Call model.dispatch() before "
            "execute_request; this function does not auto-dispatch."
        )

    fn = request_model.interventions
    tracer = request_model.tracer

    try:
        prompts, params, lora_requests, _kwargs = _compile_invokes(
            model, tracer, fn,
        )
    except Exception as e:
        logger.exception("Trace compilation failed")
        raise TraceCompilationError(f"Trace compilation failed: {e}") from e

    engine = model.vllm_entrypoint

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
        saves = {}
        for r in results:
            if r is not None:
                saves.update(pickle.loads(r))

        gen_output: GenerationOutput = {"text": "", "token_ids": []}
        if last_output is not None:
            out = last_output.outputs[0] if last_output.outputs else None
            if out is not None:
                gen_output = {
                    "text": out.text,
                    "token_ids": list(out.token_ids),
                }

        return saves, gen_output

    tasks = []
    for i, (prompt, param) in enumerate(zip(prompts, params)):
        lora_req = lora_requests[i] if lora_requests else None
        tasks.append(run_invoke(prompt, param, lora_req))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Typed envelope: per-mediator deferred user-code errors are carried
    # as a sibling list so the client can re-raise at the trace boundary
    # without rifling through ``saves``. ``__nnsight_exceptions__`` is a
    # reserved key in each invoke's saves dict (populated by
    # ``collect_nnsight``) holding ``{req_id: DeferredError}``; we pop
    # it out here before merging the real saves.
    all_saves: dict = {}
    errors: List[dict] = []
    generation_outputs: List[GenerationOutput] = []

    for result in results:
        if isinstance(result, Exception):
            logger.exception("Generation failed", exc_info=result)
            raise GenerationError(f"Generation failed: {result}") from result

        saves, gen_output = result
        nnsight_exceptions = (
            saves.pop("__nnsight_exceptions__", None) if saves else None
        )
        if nnsight_exceptions:
            for req_id, entry in nnsight_exceptions.items():
                if not isinstance(entry, dict):
                    # Defensive: legacy {type, message} shape → upgrade
                    # to the modern DeferredError envelope.
                    entry = {
                        "type_name": "Exception",
                        "message": str(entry),
                        "traceback": "",
                        "is_control_flow": False,
                    }
                entry.setdefault("req_id", req_id)
                errors.append(entry)
        all_saves.update(saves)
        generation_outputs.append(gen_output)

    return {
        "saves": all_saves,
        "errors": errors,
        "outputs": generation_outputs,
    }
